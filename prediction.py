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
    """Load the trained model from model.keras."""
    import keras  # imported here so the rest of this file works without TensorFlow

    return keras.models.load_model(str(MODEL_FILE), compile=False)


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
    # Checkpoint: the first patient in the database (Susan Foreman).
    susan = dict(zip(FEATURES, [6, 148, 72, 35, 0, 33.6, 0.627, 50]))
    model = load_model()
    print("Model input: ", make_input(susan))
    print("Prediction:  ", round(get_prediction(model, susan), 3))
    print("With BP = 90:", round(get_prediction(model, susan, blood_pressure=90), 3))
