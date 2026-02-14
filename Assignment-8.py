"""Utility for generating realistic ICU patient vitals and RAG-ready documents.

Running this script creates three one-hour CSV files (minute resolution) plus a
JSON Lines file that can be ingested by a vector database or other retrieval
pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv


load_dotenv(override=False)

DATA_DIR = Path(__file__).parent / "data"
PATIENT_SCENARIOS = [
	("P001", "sepsis"),
	("P002", "arrhythmia"),
	("P003", "respiratory_failure"),
	("P004", "stable"),
]
PATIENT_IDENTITIES: Dict[str, Dict[str, str]] = {
	"P001": {"first_name": "Layla", "last_initial": "S"},
	"P002": {"first_name": "Mateo", "last_initial": "K"},
	"P003": {"first_name": "Avery", "last_initial": "L"},
	"P004": {"first_name": "Noor", "last_initial": "H"},
}


def get_openai_api_key() -> str:
	"""Retrieve the OpenAI API key from the environment without exposing it."""

	key = os.getenv("OPENAI_API_KEY")
	if not key:
		raise RuntimeError(
			"OPENAI_API_KEY is not set. Create a .env file (excluded from git) and add the key."
		)
	return key


@dataclass
class PatientDocument:
	patient_id: str
	scenario: str
	csv_path: Path
	text: str
	alerts: List[str]
	first_name: str
	last_initial: str

	def to_jsonl(self) -> str:
		payload = {
			"patient_id": self.patient_id,
			"scenario": self.scenario,
			"source_csv": str(self.csv_path.name),
			"alerts": self.alerts,
			"patient_first_name": self.first_name,
			"patient_last_initial": self.last_initial,
			"document_text": self.text,
		}
		return json.dumps(payload, ensure_ascii=False)


def _baseline_vitals(rng: np.random.Generator) -> Dict[str, float]:
	"""Generate a slightly noisy baseline vital snapshot."""
	return {
		"ECG": "Normal Sinus Rhythm",
		"heart_rate_bpm": float(np.clip(rng.normal(76, 3), 65, 90)),
		"temperature_c": float(np.clip(rng.normal(36.7, 0.15), 36.2, 37.2)),
		"bp_systolic_mmHg": float(np.clip(rng.normal(118, 4), 108, 128)),
		"bp_diastolic_mmHg": float(np.clip(rng.normal(78, 3), 68, 84)),
		"spo2_percent": float(np.clip(rng.normal(98, 0.4), 96.5, 99.2)),
	}


def _apply_sepsis(minute: int, vitals: Dict[str, float]) -> None:
	if minute < 15:
		return
	delta = minute - 15
	vitals["temperature_c"] = min(40.2, 37.8 + 0.04 * delta)
	vitals["heart_rate_bpm"] = min(170, 95 + 1.2 * delta)
	vitals["bp_systolic_mmHg"] = max(82, vitals["bp_systolic_mmHg"] - 0.9 * delta)
	vitals["bp_diastolic_mmHg"] = max(48, vitals["bp_diastolic_mmHg"] - 0.7 * delta)
	vitals["spo2_percent"] = max(88, vitals["spo2_percent"] - 0.1 * delta)
	vitals["ECG"] = "Sinus Tachycardia"


def _apply_arrhythmia(minute: int, vitals: Dict[str, float], rng: np.random.Generator) -> None:
	if 30 <= minute <= 44:
		vitals["heart_rate_bpm"] = float(np.clip(rng.normal(175, 6), 160, 190))
		vitals["bp_systolic_mmHg"] = 88
		vitals["bp_diastolic_mmHg"] = 55
		vitals["spo2_percent"] = max(90, vitals["spo2_percent"] - 4)
		vitals["ECG"] = "Sustained Ventricular Tachycardia"
	elif minute > 44:
		vitals["heart_rate_bpm"] = float(np.clip(rng.normal(115, 4), 108, 130))
		vitals["bp_systolic_mmHg"] = float(np.clip(rng.normal(105, 5), 95, 118))
		vitals["bp_diastolic_mmHg"] = float(np.clip(rng.normal(68, 4), 60, 78))
		vitals["ECG"] = "Post-VT Sinus Tachycardia"


def _apply_respiratory(minute: int, vitals: Dict[str, float]) -> None:
	vitals["spo2_percent"] = max(81, 98 - 0.28 * minute)
	vitals["heart_rate_bpm"] = min(140, vitals["heart_rate_bpm"] + 0.6 * minute)
	if vitals["spo2_percent"] < 90:
		vitals["ECG"] = "Sinus Tachycardia with ST Depression"
		vitals["bp_systolic_mmHg"] = max(90, vitals["bp_systolic_mmHg"] - 0.2 * minute)


def _apply_stable(minute: int, vitals: Dict[str, float], rng: np.random.Generator) -> None:
	"""Keep vitals within normal bounds while adding subtle circadian-like drift."""

	if minute % 15 == 0:
		vitals["temperature_c"] += rng.normal(0, 0.05)
		vitals["heart_rate_bpm"] += rng.normal(0, 1)
		vitals["bp_systolic_mmHg"] += rng.normal(0, 1.5)
		vitals["bp_diastolic_mmHg"] += rng.normal(0, 1)
		vitals["spo2_percent"] = min(99.2, max(96.5, vitals["spo2_percent"] + rng.normal(0, 0.1)))

	if minute % 20 == 0:
		vitals["ECG"] = "Normal Sinus Rhythm"


def generate_patient_data(
	patient_id: str,
	scenario: str,
	minutes: int = 60,
	start_time: datetime | None = None,
	seed: int | None = None,
) -> pd.DataFrame:
	"""Create a DataFrame with minute-by-minute ICU vitals for one patient."""

	rng = np.random.default_rng(seed)
	start = (start_time or datetime.now()).replace(second=0, microsecond=0)
	rows: List[List[object]] = []
	identity = PATIENT_IDENTITIES.get(
		patient_id,
		{"first_name": "Patient", "last_initial": patient_id[:1] if patient_id else ""},
	)
	first_name = identity["first_name"]
	last_initial = identity["last_initial"]

	for minute in range(minutes):
		timestamp = start + timedelta(minutes=minute)
		vitals = _baseline_vitals(rng)

		if scenario == "sepsis":
			_apply_sepsis(minute, vitals)
		elif scenario == "arrhythmia":
			_apply_arrhythmia(minute, vitals, rng)
		elif scenario == "respiratory_failure":
			_apply_respiratory(minute, vitals)
		elif scenario == "stable":
			_apply_stable(minute, vitals, rng)
		else:
			raise ValueError(f"Unsupported scenario: {scenario}")

		rows.append(
			[
				patient_id,
				first_name,
				last_initial,
				timestamp.strftime("%Y-%m-%d %H:%M"),
				vitals["ECG"],
				round(vitals["heart_rate_bpm"]),
				round(vitals["temperature_c"], 1),
				round(vitals["bp_systolic_mmHg"]),
				round(vitals["bp_diastolic_mmHg"]),
				round(vitals["spo2_percent"], 1),
			]
		)

	columns = [
		"patient_id",
		"patient_first_name",
		"patient_last_initial",
		"timestamp",
		"ECG",
		"heart_rate_bpm",
		"temperature_c",
		"bp_systolic_mmHg",
		"bp_diastolic_mmHg",
		"spo2_percent",
	]
	return pd.DataFrame(rows, columns=columns)


def detect_alerts(df: pd.DataFrame) -> List[str]:
	alerts: List[str] = []
	if (df["heart_rate_bpm"] >= 150).any():
		alerts.append("Sustained tachycardia above 150 bpm")
	if (df["bp_systolic_mmHg"] <= 90).any():
		alerts.append("Hypotension detected (systolic <= 90 mmHg)")
	if (df["spo2_percent"] < 90).any():
		alerts.append("SpO2 below 90% indicating respiratory compromise")
	if (df["temperature_c"] >= 38.5).any():
		alerts.append("High-grade fever consistent with sepsis")
	if (df["ECG"].str.contains("Ventricular Tachycardia")).any():
		alerts.append("Ventricular tachycardia episode identified")
	return alerts


def build_patient_document(
	patient_id: str,
	scenario: str,
	df: pd.DataFrame,
	csv_path: Path,
	alerts: List[str],
) -> PatientDocument:
	window = f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
	mean_hr = df["heart_rate_bpm"].mean()
	min_spo2 = df["spo2_percent"].min()
	min_bp = df["bp_systolic_mmHg"].min()
	max_temp = df["temperature_c"].max()
	first_name = str(df["patient_first_name"].iloc[0]) if "patient_first_name" in df else "Patient"
	last_initial = str(df["patient_last_initial"].iloc[0]) if "patient_last_initial" in df else ""
	display_name = f"{first_name} {last_initial}." if last_initial else first_name
	text_lines = [
		f"{display_name} (patient ID {patient_id}) monitored for {window} in the {scenario.replace('_', ' ')} scenario.",
		f"Average heart rate {mean_hr:.1f} bpm, lowest SpO2 {min_spo2:.1f}%, minimum systolic BP {min_bp} mmHg, peak temperature {max_temp:.1f}C.",
	]
	if alerts:
		text_lines.append("Alerts: " + "; ".join(alerts))
	else:
		text_lines.append("No alerts triggered; patient remained stable.")

	return PatientDocument(
		patient_id=patient_id,
		scenario=scenario,
		csv_path=csv_path,
		text=" ".join(text_lines),
		alerts=alerts,
		first_name=first_name,
		last_initial=last_initial,
	)


def main() -> None:
	# Ensure the API key is available for any downstream LLM/RAG steps.
	get_openai_api_key()

	DATA_DIR.mkdir(parents=True, exist_ok=True)
	documents: List[PatientDocument] = []

	for idx, (patient_id, scenario) in enumerate(PATIENT_SCENARIOS, start=1):
		df = generate_patient_data(patient_id, scenario, seed=42 + idx)
		csv_path = DATA_DIR / f"{patient_id}_{scenario}.csv"
		df.to_csv(csv_path, index=False)
		alerts = detect_alerts(df)
		documents.append(build_patient_document(patient_id, scenario, df, csv_path, alerts))

	rag_path = DATA_DIR / "patient_rag_documents.jsonl"
	with rag_path.open("w", encoding="utf-8") as handle:
		for doc in documents:
			handle.write(doc.to_jsonl())
			handle.write("\n")

	print("Generated files:")
	for doc in documents:
		print(f" - {doc.csv_path.name} ({len(doc.alerts)} alerts)")
	print(f" - {rag_path.name} (RAG companion file)")


if __name__ == "__main__":
	main()
