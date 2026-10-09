"""Train the diabetes prediction model and save it to model.keras.

Run:  python train_model.py

This is (more or less) the model from this tutorial:
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

Data: the Pima Indians Diabetes dataset (768 patients). Each row has
8 inputs and 1 outcome (1 = diabetes within 5 years, 0 = no diabetes):
  pregnancies, glucose, diastolic blood pressure, skin thickness (mm),
  insulin, BMI, diabetes pedigree function, age, outcome
Original study: Smith et al. (1988), https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/
"""
from pathlib import Path

import numpy as np
import keras
from keras import layers

HERE = Path(__file__).parent
MODEL_FILE = HERE / "model.keras"
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# If downloading fails, save the CSV from DATA_URL next to this script under this name:
LOCAL_COPY = HERE / "pima-indians-diabetes.data.csv"

# Same seed -> same starting weights -> (almost) the same model every time.
keras.utils.set_random_seed(42)

# 1. Load the data and split it into inputs (X) and outcome (y).
source = LOCAL_COPY if LOCAL_COPY.exists() else DATA_URL
dataset = np.loadtxt(source, delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]

# 2. Define the network: 8 inputs -> 12 -> 8 -> 1 output between 0 and 1.
model = keras.Sequential([
    keras.Input(shape=(8,)),
    layers.Dense(12, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 3. Train it.
model.fit(X, y, epochs=150, batch_size=10, verbose=2)

# 4. Evaluate it. NOTE: this is measured on the same data the model was
#    trained on, so it is too optimistic. (See "Things to think about" in the README.)
_, accuracy = model.evaluate(X, y, verbose=0)
print(f"Accuracy on the training data: {accuracy * 100:.1f}%")

# 5. Quick check: a prediction for the first patient in the data.
first_patient = X[:1]
print("Prediction for the first patient:", float(model.predict(first_patient, verbose=0)[0, 0]))

# 6. Save the whole model (architecture + weights) in one file.
model.save(str(MODEL_FILE))
print(f"Saved the model to {MODEL_FILE.name}")
