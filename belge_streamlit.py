# app.py
import os
import json
import requests
import streamlit as st

st.set_page_config(page_title="BelgeNavi — Demo", page_icon="🗂️", layout="centered")

# ----- Sidebar config -----
st.sidebar.header("Backend")
BACKEND_URL = st.sidebar.text_input(
    "BelgeNavi API base URL",
    value=os.environ.get("BELGENAVI_API", "http://127.0.0.1:8000"),
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
st.title("Multi-agent BelgeNavi — Ask")
st.write("Enter your question about Turkish administrative procedures (e-İkamet, address, company, etc.).")

lang = st.selectbox(
    "Language",
    options=["auto", "en", "ar", "tr"],
    index=0,
    help="Let the backend auto-detect, or force a language."
)

query = st.text_area(
    "Your question",
    placeholder="Example: I want to renew my short-term residence permit in İstanbul. What documents do I need?",
    height=140,
)

if st.button("Ask BelgeNavi", type="primary", disabled=not query.strip()):
    try:
        payload = {"query": query.strip(), "lang": lang}
        with st.spinner("Contacting BelgeNavi…"):
            resp = requests.post(f"{BACKEND_URL}/ask", json=payload)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text[:300]}")
        else:
            data = resp.json()

            # Try to present nicely if your graph returns structured parts
            # Fallback to raw JSON if unknown
            st.success("Answer received")

            # Common sections you might have (adjust to your graph’s output)
            if isinstance(data, dict):
                # Show known sections in expanders if present
                for key in ["classifier", "retriever", "citer", "checklist_composer", "form_filler", "guardrails", "summary"]:
                    if key in data:
                        with st.expander(key.upper(), expanded=(key in ["citer","summary"])):
                            st.json(data[key])
                # If none matched, show full object
                if not any(k in data for k in ["classifier","retriever","citer","checklist_composer","form_filler","guardrails","summary"]):
                    st.json(data)
            else:
                st.json(data)

            # If your API returns an ICS string or file content, offer a download
            # Example: data.get("ics") could be a string
            ics_text = None
            if isinstance(data, dict):
                ics_text = data.get("ics") or (data.get("schedule") if isinstance(data.get("schedule"), str) else None)
            if ics_text:
                st.download_button("Download reminder (.ics)", data=ics_text, file_name="belgenavi-reminder.ics", mime="text/calendar")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except json.JSONDecodeError:
        st.error("Response was not valid JSON.")
