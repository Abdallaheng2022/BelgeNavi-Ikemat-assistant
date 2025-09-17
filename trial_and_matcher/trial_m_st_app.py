import os
import json
import requests
import streamlit as st

st.set_page_config(page_title="TrialMatcher — Demo", page_icon="🧪", layout="centered")

# ----- Sidebar config -----
st.sidebar.header("Backend")
BACKEND_URL = st.sidebar.text_input(
    "TrialMatcher API base URL",
    value=os.environ.get("TRIALMATCHER_API", "http://127.0.0.1:8000"),
    help="Your FastAPI server root. Example: http://127.0.0.1:8000",
)

# Optional quick health check
def check_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

healthy = check_health()
st.sidebar.markdown(f"**API status:** {'🟢 online' if healthy else '🔴 offline?'}")

# ----- Main UI -----
st.title("Multi-agent TrialMatcher — Clinical Trial Assistant")
st.write("Ask about patient context and trial matching. The backend will run explainer → validator → summarizer steps.")

lang = st.selectbox(
    "Language",
    options=["auto", "en", "ar", "tr"],
    index=0,
    help="Let the backend auto-detect, or force a language."
)

query = st.text_area(
    "Enter patient context or question",
    placeholder="Example: A 45-year-old male with diabetes looking for oncology trials in İstanbul",
    height=140,
)

st.text(query)
if st.button("Run TrialMatcher", type="primary", disabled=not query.strip()):
    try:
        payload = {"query": query.strip(),"lang":lang}
        with st.spinner("Contacting TrialMatcher…"):
            resp = requests.post(f"{BACKEND_URL}/match",headers = {"Content-Type": "application/json"}, json=payload)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text[:300]}")
        else:
            data = resp.json()
            st.success("Results received")

            if isinstance(data, dict):
                # Show pipeline steps (if backend returns them)
                for key in ["retriever", "explainer", "validator","summarize"]:
                    if key in data:
                        with st.expander(key.upper(), expanded=(key in ["summarize"])):
                            st.json(data[key])

                # If nothing matched, show full JSON
                if not any(k in data for k in ["explainer", "validator", "summarize"]):
                    st.json(data)
            else:
                st.json(data)

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except json.JSONDecodeError:
        st.error("Response was not valid JSON.")