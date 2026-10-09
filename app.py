"""A toy clinical decision support app.

Run:  streamlit run app.py

How Streamlit works, in one sentence: every time the user clicks or types
something, Streamlit runs this whole script again from top to bottom and
redraws the page. Things that must survive between runs (like a value the
user typed) are kept in st.session_state.

This is a technical demo. It is NOT meant to be used for clinical decisions.
"""
import sqlite3
from contextlib import closing
from pathlib import Path

import streamlit as st

import prediction

DB_FILE = Path(__file__).parent / "ehr.db"

st.set_page_config(page_title="A simple ML demo")
st.title("A simple ML demo")
st.caption("Technical demo only. Not intended for clinical decisions.")

# --- 1. Check that the files we need are there ------------------------------
if not DB_FILE.exists():
    st.error("ehr.db not found. Run `python setup_database.py` first (see the README).")
    st.stop()
if not prediction.MODEL_FILE.exists():
    st.error("model.keras not found. Run `python train_model.py` first (see the README).")
    st.stop()


# --- 2. Load the model (once) and the patients (on every run) ---------------
@st.cache_resource  # load the model only once, not on every click
def get_model():
    return prediction.load_model()


def load_patients():
    """Read all patients from the database, as {id: {column: value}}."""
    with closing(sqlite3.connect(DB_FILE)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM patient ORDER BY id").fetchall()
    return {row["id"]: dict(row) for row in rows}


model = get_model()
patients = load_patients()
if not patients:
    st.warning("The database has no patients. Check database.sql and run setup_database.py.")
    st.stop()

# The blood pressure typed in by the user. None = use the value from the database.
st.session_state.setdefault("userbp", None)


def forget_userbp():
    st.session_state.userbp = None


# --- 3. The page -------------------------------------------------------------
st.subheader("Patients")
patient_id = st.radio(
    "Select a patient",
    options=list(patients),
    format_func=lambda pid: patients[pid]["display_name"],
    on_change=forget_userbp,  # a new patient starts without a typed-in value
)
current_patient = patients[patient_id]

results = st.container()  # filled in below, but shown here, above the form

with st.form("bp_form", clear_on_submit=True):
    new_bp = st.number_input(
        "If desired, enter a new blood pressure value here",
        min_value=1, max_value=400, value=None, step=1,
        help="Leave empty and press Submit to go back to the value in the database.",
    )
    if st.form_submit_button("Submit"):
        st.session_state.userbp = new_bp  # used for this prediction only, not saved

with results:
    userbp = st.session_state.userbp
    p = prediction.get_prediction(model, current_patient, userbp)

    st.markdown(f"**Current patient:** {current_patient['display_name']}")
    st.markdown(f"**User-entered blood pressure:** {userbp if userbp is not None else 'none'}")
    st.metric("Prediction", f"{p:.2f}")
    st.caption("The model's output, between 0 and 1. Higher means the model thinks "
               "diabetes is more likely. It is not a calibrated risk.")

    with st.expander("What the model receives"):
        x = prediction.make_input(current_patient, userbp)[0]
        st.table({"input": prediction.FEATURES, "value": [float(v) for v in x]})
