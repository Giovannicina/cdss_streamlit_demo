"""Everything to do with the model: loading it and making a prediction.

You can test this file on its own, without the web app:
    python prediction.py
"""
from pathlib import Path

import numpy as np

MODEL_FILE = Path(__file__).parent / "model.keras"

# The model expects its 8 inputs in exactly this order (the column order of
# the training data). These are also the column names in the database.
FEATURES = [
    "num_pregnancies",
    "glucose",
    "blood_pressure_dias",
    "skin_thickness",
    "insulin",
    "BMI",
    "diab_pedigree",
    "age",
]


def load_model():
    """Load your trained model from model.keras."""
    import keras  # imported here so the rest of this file works without TensorFlow

    return keras.models.load_model(str(MODEL_FILE), compile=False)


def check_model(model, X):
    """Check that the model fits this app. Returns a problem description, or None if OK.

    X holds some example inputs, one row of 8 values per patient.
    """
    try:
        out = np.asarray(model.predict(X, verbose=0))
    except Exception as error:
        n = len(FEATURES)
        return (f"The model could not make a prediction from {n} input values. "
                f"Does it expect exactly {n} inputs, one for each name in FEATURES? (Error: {error})")
    if out.shape != (len(X), 1):
        return (f"The model should return one number per patient, i.e. an output of shape "
                f"({len(X)}, 1) for {len(X)} patients, but it returned shape {out.shape}.")
    if not np.all((out >= 0) & (out <= 1)):
        return ("The model returned values outside the range 0 to 1, "
                "so its output cannot be read as a probability.")
    return None


def make_input(patient, blood_pressure=None):
    """Turn a patient (a dict with the FEATURES as keys) into the model's input.

    If blood_pressure is given, it replaces the patient's stored value
    for this prediction only (nothing is written to the database).
    """
    values = dict(patient)
    if blood_pressure is not None:
        values["blood_pressure_dias"] = blood_pressure
    return np.array([[float(values[name]) for name in FEATURES]])


def get_prediction(model, patient, blood_pressure=None):
    """Return the model's output for one patient: a number between 0 and 1."""
    x = make_input(patient, blood_pressure)
    return float(model.predict(x, verbose=0)[0, 0])


if __name__ == "__main__":
    # Checkpoint: the five patients from database.sql, the first one being Susan Foreman.
    examples = np.array([
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],
        [2, 106, 56, 27, 165, 29.0, 0.426, 22],
        [2, 174, 88, 37, 120, 44.5, 0.646, 24],
        [4, 95, 60, 32, 0, 35.4, 0.284, 28],
        [0, 126, 86, 27, 120, 27.4, 0.515, 21],
    ], dtype=float)
    susan = dict(zip(FEATURES, examples[0]))
    model = load_model()
    problem = check_model(model, examples)
    if problem:
        raise SystemExit("Problem with model.keras: " + problem)
    print("Model input: ", make_input(susan))
    print("Prediction:  ", round(get_prediction(model, susan), 3))
    print("With BP = 90:", round(get_prediction(model, susan, blood_pressure=90), 3))
