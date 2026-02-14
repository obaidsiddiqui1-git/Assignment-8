"""Streamlit UI for browsing local ICU RAG files and generating diagnoses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from textwrap import dedent
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=False)

DATA_DIR = Path(__file__).parent / "data"
OPENAI_MODEL = "gpt-4o-mini"
STABLE_LABEL = "No critical deterioration detected"
REQUIRED_COLUMNS = [
    "patient_id",
    "timestamp",
    "ECG",
    "heart_rate_bpm",
    "temperature_c",
    "bp_systolic_mmHg",
    "bp_diastolic_mmHg",
    "spo2_percent",
]
OPTIONAL_NAME_COLUMNS = {
    "patient_first_name": "Patient",
    "patient_last_initial": "",
}

if "ai_last_telemetry" not in st.session_state:
    st.session_state["ai_last_telemetry"] = None


def detect_alerts(df: pd.DataFrame) -> List[Tuple[str, str]]:
    alerts: List[Tuple[str, str]] = []
    if (df["heart_rate_bpm"] >= 150).any():
        alerts.append(("Severe tachycardia", "Sustained HR ≥150 bpm"))
    if (df["bp_systolic_mmHg"] <= 90).any():
        alerts.append(("Hypotension", "Systolic BP ≤90 mmHg"))
    if (df["spo2_percent"] < 90).any():
        alerts.append(("Hypoxia", "SpO₂ <90%"))
    if (df["temperature_c"] >= 38.5).any():
        alerts.append(("High-grade fever", "Temperature ≥38.5°C"))
    if (df["ECG"].str.contains("ventricular tachycardia", case=False, na=False)).any():
        alerts.append(("Ventricular tachycardia", "ECG indicates VT"))
    return alerts


def classify_condition(df: pd.DataFrame) -> Tuple[str, str]:
    tachy_hr = df["heart_rate_bpm"].median()
    temp_max = df["temperature_c"].max()
    spo2_min = df["spo2_percent"].min()
    sys_min = df["bp_systolic_mmHg"].min()
    vtach = (df["ECG"].str.contains("ventricular tachycardia", case=False, na=False)).any()

    if vtach or tachy_hr >= 160:
        return (
            "Arrhythmia / Ventricular Tachycardia",
            "Telemetry shows sustained VT or extreme HR elevations that demand immediate ACLS protocol readiness.",
        )
    if temp_max >= 38.5 and sys_min <= 95 and tachy_hr >= 110:
        return (
            "Sepis / distributive shock trend",
            "Fever, hypotension, and tachycardia point to sepsis progression. Begin sepsis bundle and fluid resuscitation.",
        )
    if spo2_min < 90:
        return (
            "Respiratory failure progression",
            "SpO₂ drifted under 90% despite compensatory tachycardia, suggesting escalating respiratory compromise.",
        )
    return (
        STABLE_LABEL,
        "Vitals stay within expected ranges; continue routine monitoring and trending.",
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    df = df.copy()
    for col, default in OPTIONAL_NAME_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    return df


def list_patient_files() -> List[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(DATA_DIR.glob("*.csv"))


def regenerate_rag_data() -> None:
    try:
        result = subprocess.run(
            [sys.executable, "Assignment-8.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        st.session_state["generator_log"] = result.stdout.strip() or "Generator finished."
    except subprocess.CalledProcessError as exc:
        st.session_state["generator_log"] = exc.stderr.strip() or str(exc)


def prime_ai_telemetry(patient_files: List[Path]) -> None:
    if st.session_state.get("ai_last_telemetry"):
        return
    for patient_path in patient_files:
        try:
            df = load_dataframe(patient_path)
        except Exception:
            continue
        alerts = detect_alerts(df)
        label, _ = classify_condition(df)
        if not alerts and label == STABLE_LABEL:
            continue
        patient_id = df["patient_id"].iloc[0]
        scenario = (
            patient_path.stem.split("_", 1)[1] if "_" in patient_path.stem else "unknown"
        )
        snapshot = build_patient_snapshot(patient_id, scenario, label, alerts, df)
        try:
            generate_agentic_plan(snapshot)
        except Exception:
            continue
        else:
            break


def check_vitals_now() -> None:
    regenerate_rag_data()
    st.session_state.pop("ai_last_telemetry", None)
    new_files = list_patient_files()
    prime_ai_telemetry(new_files)
    st.rerun()


def build_patient_snapshot(
    patient_id: str,
    scenario: str,
    label: str,
    alerts: List[Tuple[str, str]],
    df: pd.DataFrame,
) -> Dict[str, object]:
    recent = df.tail(5).copy()
    if "timestamp" in recent:
        recent["timestamp"] = recent["timestamp"].apply(
            lambda ts: ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        )

    start_ts = df["timestamp"].iloc[0] if not df.empty else None
    end_ts = df["timestamp"].iloc[-1] if not df.empty else None
    def _to_iso(value):
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    start_str = _to_iso(start_ts)
    end_str = _to_iso(end_ts)

    return {
        "patient_id": patient_id,
        "scenario": scenario,
        "condition_label": label,
        "alerts": alerts,
        "window": {
            "start": start_str,
            "end": end_str,
        },
        "heart_rate": {
            "mean": float(df["heart_rate_bpm"].mean()),
            "max": float(df["heart_rate_bpm"].max()),
            "min": float(df["heart_rate_bpm"].min()),
        },
        "blood_pressure": {
            "systolic_min": float(df["bp_systolic_mmHg"].min()),
            "diastolic_min": float(df["bp_diastolic_mmHg"].min()),
        },
        "spo2_min": float(df["spo2_percent"].min()),
        "temperature_max": float(df["temperature_c"].max()),
        "recent_samples": recent.to_dict(orient="records"),
    }


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file or environment before running the app."
        )
    return OpenAI(api_key=key)


def update_ai_telemetry(prompt_tokens: int | None, completion_tokens: int | None, latency_seconds: float) -> None:
    input_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
    output_tokens = int(completion_tokens) if completion_tokens is not None else 0
    st.session_state["ai_last_telemetry"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": round(latency_seconds * 1000, 1),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def generate_agentic_plan(snapshot: Dict[str, object]) -> Dict[str, object]:
    client = get_openai_client()
    snapshot_json = json.dumps(snapshot, ensure_ascii=False)
    system_prompt = (
        "You are ICU Rapid Response Copilot, an autonomous agent that triages deteriorating patients. "
        "Provide concise, protocol-aware directives for bedside nurses."
    )
    user_prompt = f"""
Analyze the ICU snapshot below and respond with JSON matching this schema:
{{
  "urgency": "emergency" | "urgent" | "routine",
  "summary": "Justification for the urgency",
  "verification_steps": ["Steps to double-check monitors/lines"],
  "actions": ["Immediate nurse actions (mention rapid response/code triggers when needed)"],
  "notifications": ["Teams or clinicians to notify"],
  "medications": ["Optional drug/infusion suggestions with brief dosing cues"],
  "watchouts": ["Potential complications to monitor"]
}}

Rules:
- Always include at least one verification step.
- Mention notifying clinicians/pharmacy/respiratory therapy when appropriate.
- Recommend lifesaving drugs only when indicated and cite guideline anchors (ACLS, Surviving Sepsis, etc.).
- If ventricular tachycardia, shock, or SpO2 < 88%, set urgency="emergency" and reference rapid response/code activation.
- Respond with JSON only.

Snapshot:
{snapshot_json}
"""

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    latency = time.perf_counter() - start_time
    usage = getattr(response, "usage", None)
    prompt_tokens = None
    completion_tokens = None
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None and hasattr(usage, "get"):
            prompt_tokens = usage.get("prompt_tokens")
        if completion_tokens is None and hasattr(usage, "get"):
            completion_tokens = usage.get("completion_tokens")
    update_ai_telemetry(prompt_tokens, completion_tokens, latency)
    content = response.choices[0].message.content
    return json.loads(content)


def render_patient_card(patient_path: Path) -> None:
    try:
        patient_df = load_dataframe(patient_path)
    except Exception as exc:
        st.error(f"Failed to load {patient_path.name}: {exc}")
        return

    patient_id = patient_df["patient_id"].iloc[0]
    scenario = patient_path.stem.split("_", 1)[1] if "_" in patient_path.stem else "unknown"
    scenario_title = scenario.replace("_", " ").title()
    first_name = (
        str(patient_df["patient_first_name"].iloc[0])
        if "patient_first_name" in patient_df
        else "Patient"
    )
    last_initial = (
        str(patient_df["patient_last_initial"].iloc[0])
        if "patient_last_initial" in patient_df
        else ""
    )
    display_name = f"{first_name} {last_initial}." if last_initial else first_name

    card = st.container()
    with card:
        st.markdown("<span class='patient-card-anchor'></span>", unsafe_allow_html=True)
        header_html = f"""
        <div class='patient-card-header'>
            <div>
                <h4>Patient {patient_id} · {scenario_title}</h4>
                <small>Window: {patient_df['timestamp'].iloc[0]} — {patient_df['timestamp'].iloc[-1]}</small>
            </div>
            <div class='patient-name-tag'>{display_name}</div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

        metric_block = """
        <div class='metric-grid'>
            <div><span class='metric-label'>Avg HR</span><span class='metric-value'>{avg_hr} bpm</span></div>
            <div><span class='metric-label'>Min SBP</span><span class='metric-value'>{min_sbp} mmHg</span></div>
            <div><span class='metric-label'>Min SpO₂</span><span class='metric-value'>{min_spo2}%</span></div>
            <div><span class='metric-label'>Max Temp</span><span class='metric-value'>{max_temp}°C</span></div>
        </div>
        """
        st.markdown(
            metric_block.format(
                avg_hr=f"{patient_df['heart_rate_bpm'].mean():.0f}",
                min_sbp=f"{patient_df['bp_systolic_mmHg'].min():.0f}",
                min_spo2=f"{patient_df['spo2_percent'].min():.1f}",
                max_temp=f"{patient_df['temperature_c'].max():.1f}",
            ),
            unsafe_allow_html=True,
        )

        body = st.container()
        with body:
            st.markdown("<span class='patient-card-body-anchor'></span>", unsafe_allow_html=True)

            alerts = detect_alerts(patient_df)
            label, rationale = classify_condition(patient_df)

            abnormal = bool(alerts) or label != STABLE_LABEL
            plan: Dict[str, object] | None = None
            if abnormal:
                snapshot = build_patient_snapshot(patient_id, scenario, label, alerts, patient_df)
                try:
                    plan = generate_agentic_plan(snapshot)
                except Exception as exc:
                    st.error(f"Agentic planner error for {patient_id}: {exc}")

            urgency = (plan or {}).get("urgency", "routine").lower()
            urgency_summary = (plan or {}).get(
                "summary",
                "No emergency triggers detected." if not abnormal else "Observation-only alerts; continue monitoring.",
            )

            emergency_col, alert_col = st.columns([2, 3])
            with emergency_col:
                st.markdown("**Emergency**")
                if plan:
                    urgency_class = {
                        "emergency": "emergency",
                        "urgent": "urgent",
                        "routine": "routine",
                    }.get(urgency, "routine")
                    icon = {"emergency": "🚨", "urgent": "⚠️", "routine": "ℹ️"}.get(
                        urgency, "ℹ️"
                    )
                    st.markdown(
                        f"<div class='urgency-pill {urgency_class}'>{icon} {urgency_summary}</div>",
                        unsafe_allow_html=True,
                    )
                elif abnormal:
                    st.markdown(
                        "<div class='urgency-pill urgent'>⚠️ Abnormal vitals detected; planner unavailable.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='urgency-pill routine'>✅ Stable vitals; no emergency at this time.</div>",
                        unsafe_allow_html=True,
                    )

            with alert_col:
                st.markdown("**Alert**")
                if alerts:
                    for title, detail in alerts:
                        st.markdown(
                            f"<div class='observation-pill alert'><strong>{title}:</strong> {detail}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<div class='observation-pill ok'>No immediate alerts triggered.</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("**Diagnosis**")
            st.write(f"{label} — {rationale}")

            if plan:
                if plan.get("verification_steps"):
                    st.markdown("**Verify / Stabilize**")
                    for step in plan["verification_steps"]:
                        st.write(f"- {step}")
                if plan.get("actions"):
                    st.markdown("**Immediate Actions**")
                    for action in plan["actions"]:
                        st.write(f"- {action}")
                if plan.get("notifications"):
                    st.markdown("**Notifications / Escalations**")
                    for note in plan["notifications"]:
                        st.write(f"- {note}")
                if plan.get("medications"):
                    st.markdown("**Medication / Drip Guidance**")
                    for med in plan["medications"]:
                        st.write(f"- {med}")
                if plan.get("watchouts"):
                    st.markdown("**Watch-outs**")
                    for warning in plan["watchouts"]:
                        st.write(f"- {warning}")
            elif abnormal:
                st.warning(
                    f"Abnormal vitals detected for {patient_id} but agentic planner could not generate guidance."
                )
            else:
                st.info("Vitals currently within acceptable range; no agentic intervention required.")

            with st.expander("Vitals trends & raw data"):
                trend_cols = st.columns(2)
                with trend_cols[0]:
                    st.line_chart(
                        patient_df.set_index("timestamp")[
                            ["heart_rate_bpm", "bp_systolic_mmHg", "bp_diastolic_mmHg"]
                        ]
                    )
                with trend_cols[1]:
                    st.line_chart(
                        patient_df.set_index("timestamp")[
                            ["spo2_percent", "temperature_c"]
                        ]
                    )
                st.dataframe(patient_df.tail(20), width="stretch", height=140)


st.set_page_config(page_title="AI Patient Monitor", layout="wide")
st.markdown(
    """
    <style>
    .flash-alert {
        background-color: #b20320;
        color: #fff;
        font-weight: 700;
        padding: 1rem;
        border-radius: 0.6rem;
        text-align: center;
        animation: flash 1s linear infinite;
        margin-bottom: 1rem;
    }
    @keyframes flash {
        0% { opacity: 1; }
        50% { opacity: 0.15; }
        100% { opacity: 1; }
    }
    .patient-card-anchor,
    .patient-card-body-anchor {
        display: none;
    }
    div[data-testid="stVerticalBlock"]:has(.patient-card-anchor) {
        border: 1px solid #1f2937;
        border-radius: 0.8rem;
        padding: 0.55rem 0.95rem 0.85rem;
        height: 560px;
        display: flex;
        flex-direction: column;
        background: #0f1116;
        color: #dbeafe;
        box-shadow: 0 0 20px rgba(15,17,22,0.45);
    }
    div[data-testid="stVerticalBlock"]:has(.patient-card-anchor) h4 {
        margin: 0;
        font-size: 1.05rem;
    }
    div[data-testid="stVerticalBlock"]:has(.patient-card-anchor) small {
        color: #9ca3af;
        display: block;
        margin-top: 0.15rem;
    }
    div[data-testid="stVerticalBlock"]:has(.patient-card-anchor)
    div[data-testid="stVerticalBlock"]:has(.patient-card-body-anchor) {
        margin-top: 0.6rem;
        flex: 1;
        overflow-y: auto;
        padding-right: 0.35rem;
    }
    .patient-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.3rem;
    }
    .patient-name-tag {
        font-size: 0.9rem;
        font-weight: 600;
        color: #bfdbfe;
        background: rgba(59,130,246,0.2);
        border: 1px solid rgba(59,130,246,0.45);
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        white-space: nowrap;
    }
    .hospital-hero {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .hero-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .hospital-logo {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        position: relative;
        background: radial-gradient(circle at 35% 40%, #bfdbfe 0%, #2563eb 55%, #0f172a 100%);
        box-shadow: 0 0 25px rgba(59,130,246,0.65);
    }
    .hospital-logo::before {
        content: "";
        position: absolute;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: #0f172a;
        top: 14px;
        left: 34px;
    }
    .hospital-title {
        color: #60a5fa;
        font-size: 2.4rem;
        margin: 0;
    }
    .hospital-subtitle {
        color: #f87171;
        margin: 0.2rem 0;
        font-size: 1.2rem;
    }
    .hero-right {
        min-width: 280px;
        max-width: 420px;
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.4rem;
        background: #0b1120;
        border: 1px solid rgba(59,130,246,0.45);
        border-radius: 0.9rem;
        padding: 0.8rem 1rem;
        box-shadow: 0 0 18px rgba(15,23,42,0.65);
    }
    .ai-telemetry-title {
        margin: 0;
        font-size: 0.95rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #93c5fd;
    }
    .ai-telemetry-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        width: 100%;
    }
    .ai-telemetry-grid div {
        background: rgba(30,58,138,0.5);
        border-radius: 0.6rem;
        padding: 0.5rem 0.6rem;
        border: 1px solid rgba(59,130,246,0.35);
    }
    .ai-telemetry-label {
        display: block;
        font-size: 0.7rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.15rem;
    }
    .ai-telemetry-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e0f2fe;
    }
    .ai-telemetry-meta {
        margin: 0;
        font-size: 0.75rem;
        color: #9ca3af;
    }
    .ai-telemetry-empty {
        width: 100%;
        border: 1px dashed rgba(59,130,246,0.4);
        border-radius: 0.8rem;
        padding: 0.65rem 0.75rem;
        color: #9ca3af;
        font-size: 0.85rem;
    }
    .observation-pill {
        background: #1f2937;
        border-radius: 0.5rem;
        padding: 0.35rem 0.6rem;
        margin-bottom: 0.35rem;
        font-size: 0.85rem;
        line-height: 1.2rem;
    }
    .observation-pill.alert {
        border-left: 4px solid #d97706;
    }
    .observation-pill.ok {
        border-left: 4px solid #10b981;
        color: #a7f3d0;
    }
    .urgency-pill {
        border-radius: 0.6rem;
        padding: 0.6rem 0.75rem;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }
    .urgency-pill.emergency {
        background: #b20320;
        color: #fff;
        box-shadow: 0 0 12px rgba(178,3,32,0.6);
        animation: flash 1s linear infinite;
    }
    .urgency-pill.urgent {
        background: #92400e;
        color: #fff;
    }
    .urgency-pill.routine {
        background: #065f46;
        color: #ecfdf5;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem 0.8rem;
        margin: 0.4rem 0 0.8rem 0;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #9ca3af;
        display: block;
    }
    .metric-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f3f4f6;
    }
    .sidebar-section-title {
        color: #60a5fa;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1rem 0 0.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

patient_files = list_patient_files()
prime_ai_telemetry(patient_files)

telemetry_snapshot = st.session_state.get("ai_last_telemetry")
if telemetry_snapshot:
    telemetry_html = dedent(
        f"""
        <div class='ai-telemetry-grid'>
            <div>
                <span class='ai-telemetry-label'>Input Tokens</span>
                <span class='ai-telemetry-value'>{telemetry_snapshot['input_tokens']}</span>
            </div>
            <div>
                <span class='ai-telemetry-label'>Output Tokens</span>
                <span class='ai-telemetry-value'>{telemetry_snapshot['output_tokens']}</span>
            </div>
            <div>
                <span class='ai-telemetry-label'>Latency</span>
                <span class='ai-telemetry-value'>{telemetry_snapshot['latency_ms']} ms</span>
            </div>
        </div>
        <p class='ai-telemetry-meta'>Last run · {telemetry_snapshot['timestamp']}</p>
        """
    ).strip()
else:
    telemetry_html = "<div class='ai-telemetry-empty'>Telemetry populates after the first agentic plan run.</div>"

hero_html = f"""
<div class='hospital-hero'>
    <div class='hero-left'>
        <div class='hospital-logo'></div>
        <div>
            <h1 class='hospital-title'>Zia and Noor ER &amp; Urgent Care - Plano</h1>
            <p class='hospital-subtitle'>AI-Based ICU Patient Monitor</p>
        </div>
    </div>
    <div class='hero-right'>
        <p class='ai-telemetry-title'>Billing: AI Observability Reporting</p>
        {telemetry_html}
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Zia and Noor ER & Urgent Care - Plano")
    st.markdown("**Address:** 1842 Meridian Ave, Austin, TX 78701")
    st.markdown("**Main Line:** (512) 555-0176")

    st.markdown('<div class="sidebar-section-title">ICU Nurse on Duty</div>', unsafe_allow_html=True)
    nurses = [
        ("Nurse Priya Kale", "(512) 555-0199"),
        ("Nurse Omar Reyes", "(512) 555-0125"),
    ]
    for name, phone in nurses:
        st.markdown(f"**{name}:** {phone}")

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">On-Call Specialists</div>', unsafe_allow_html=True)
    doctors = [
        ("Critical Care", "Dr. Maya Riaz", "(512) 555-0114"),
        ("Cardiology", "Dr. Leon Park", "(512) 555-0138"),
        ("Respiratory Therapy", "Ava Patel, RRT", "(512) 555-0182"),
        ("Pharmacy", "Dr. Noah Binh, PharmD", "(512) 555-0159"),
    ]
    for dept, name, phone in doctors:
        st.markdown(f"**{dept}:** {name} · {phone}")

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Rapid Response Tips</div>', unsafe_allow_html=True)
    st.markdown(
        "- [ACLS algorithm refresher](https://promedcert.com/blog/important-acls-algorithms-you-need-to-know)")
    st.markdown(
        "- [Managing sepsis bundles](https://www.sccm.org/survivingsepsiscampaign/guidelines-and-resources)")
    st.markdown(
        "- [Stroke code checklist](https://www.thehaugengroup.com/the-befast-checklist-for-coding-strokes/)")

    st.markdown("---")
    if st.button("Check Vitals Now", type="primary", use_container_width=True):
        check_vitals_now()

if not patient_files:
    st.warning("No CSV files detected in the RAG folder. Generate data from the sidebar to continue.")
    st.stop()

st.markdown("## Patient Dashboards")

for i in range(0, len(patient_files), 2):
    row_files = patient_files[i : i + 2]
    columns = st.columns(2)
    for col, patient_path in zip(columns, row_files):
        with col:
            render_patient_card(patient_path)
    if len(row_files) < len(columns):
        for col in columns[len(row_files) :]:
            with col:
                st.empty()
