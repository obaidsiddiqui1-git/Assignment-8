# Zia & Noor ER Agentic ICU Assistant

An AI-assisted ICU monitoring prototype that simulates realistic patient vitals, stores them as local RAG artifacts, and serves an agentic Streamlit dashboard capable of issuing rapid-response guidance and billing-friendly telemetry.

## Quick Start

```bash
pip install -r requirements.txt  # if you maintain one
streamlit run streamlit_app.py
```

Use the left sidebar to regenerate synthetic patients ("Check Vitals Now") and to review the current on-call roster. The hero header reports OpenAI token usage and latency for billing observability after each agentic call.

## AI Architecture & Workflow

The solution is split into four cooperating layers.

1. **Synthetic Patient Simulator – `Assignment-8.py`**  
   Generates per-minute vitals for four scenarios (sepsis, ventricular tachycardia, respiratory failure, stable) and injects scenario-specific drifts. Each CSV row includes friendly patient metadata plus alert summaries; a companion `patient_rag_documents.jsonl` file supports retrieval workflows.

2. **RAG Vault & Refresh Controls – `data/`**  
   The Streamlit app scans the folder dynamically, normalizes timestamps, and fills optional identity columns. "Check Vitals Now" reruns the generator, clears caches, and primes telemetry so the next OpenAI call is immediately observable.

3. **Agentic Reasoner – `generate_agentic_plan()`**  
   Patient snapshots (recent vitals, alerts, derived stats) are serialized to JSON and sent to OpenAI `gpt-4o-mini`. A strict schema enforces urgency, summary, verification steps, actions, medication/drip guidance, notifications, and watch-outs. Token usage + latency feed the billing card.

4. **ICU Command UI – `streamlit_app.py`**  
   Streamlit renders a 2×2 grid of bounded patient cards. Alerts and emergency status sit above diagnoses, with scrollable details, vitals trend charts, and RAG expander tables. The hero banner blends hospital styling with observability telemetry.

### Flowchart

```mermaid
flowchart LR
    A[Assignment-8.py
    Synthetic Scenario Generator] -->|CSV + JSONL| B[data/ RAG Vault]
    B -->|load_dataframe()| C[Streamlit Patient Cards]
    C -->|build_patient_snapshot()| D[OpenAI gpt-4o-mini]
    D -->|Agentic Plan JSON
    + usage stats| C
    C -->|prime_ai_telemetry()
    update_ai_telemetry()| E[Billing Observability]
    C -->|Check Vitals Now
    reruns Assignment-8.py| A
```

## Key Commands

- `Assignment-8.py`: Regenerates synthetic patient vitals and RAG documents.
- `streamlit run streamlit_app.py`: Launches the dashboard.
- Sidebar ➜ **Check Vitals Now**: Calls the simulator, primes telemetry, and refreshes the UI.

## Notes

- Environment variables (`OPENAI_API_KEY`, etc.) are managed via `.env` and `python-dotenv`.
- The Streamlit app expects any CSV placed in `data/` to include at least the required vitals columns; optional name columns are auto-filled if missing.
- Telemetry defaults to the first abnormal patient on load, ensuring billing stats render without manual interaction.
