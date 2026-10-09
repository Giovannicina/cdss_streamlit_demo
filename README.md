# A toy clinical decision support app, built with Streamlit

This repo contains a small demo for students: a web app that lets a user

1. pick a patient from a tiny (fictional) health record,
2. optionally type in a different blood pressure value, and
3. see what a machine-learning model predicts about that patient's chance of diabetes.

The app is ready to go, except for one thing: **the model. This must be trained by students in a separate script.**

This is a [Streamlit](https://streamlit.io) version of
[Python_ML_web_app_demo](https://github.com/ace-dvm/Python_ML_web_app_demo) (Flask + PythonAnywhere),
with the same data and the same five patients.

> ⚠️ **This is a technical demo. It must not be used to make clinical decisions.**

---

## What's in this repo

| File | What it does |
|---|---|
| `app.py` | The web app (the page you see in the browser). |
| `prediction.py` | Loads your model, checks it, and turns a patient into a prediction. |
| `database.sql` | The patient records, written in SQL. |
| `setup_database.py` | Turns `database.sql` into the database file `ehr.db`. |
| `ehr.db` | The database file (already built for you). |
| `requirements.txt` | The Python packages the app needs. |
| `model.keras` | **Not included: you train and add this yourself** (see below). |

## How the pieces fit together

```
 database.sql ──(setup_database.py)──▶ ehr.db ─────────┐
                                                       ▼
                                  app.py ◀──▶ prediction.py ◀── model.keras
                                    ▲                              ▲
                                    │                              │
                    you, in the browser                  your own training code
```

When you click something in the app, Streamlit re-runs `app.py` from top to bottom:
it reads the patients from `ehr.db`, builds the 8 numbers the model needs, asks the
model for a prediction and redraws the page.

---

## Your task: train the model

For this demo you train a neural network that predicts diabetes, using the **Pima Indians Diabetes** dataset:
[pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
(768 patients, no header row; the 9th column is the outcome: 1 = diabetes, 0 = no diabetes).
**Note: to customize this app to another problem, you need to train a custom model for that problem and data.**

You decide how to build and train it. To work for this example, your model must:

1. **Take 8 inputs, in this order:**
   pregnancies, glucose, diastolic blood pressure, skin thickness, insulin, BMI,
   diabetes pedigree function, age. This is the column order of the CSV, and the order
   of `FEATURES` in `prediction.py`.
2. **Accept the raw values** as they are in the database (e.g. glucose = 148, age = 50).
   If your model needs scaled inputs, put the scaling *inside* the model, or add it to `prediction.py`.
3. **Output one number between 0 and 1** for each patient: the model's estimate that the patient has diabetes.
4. **Be a Keras model saved as `model.keras`**, with `model.save("model.keras")`,
   and placed in the same folder as `app.py`.

The app checks points 1 and 3 when it starts and tells you if something doesn't fit.
It cannot check point 2 or the order in point 1, so double-check those yourself:
**if the order or scaling is wrong, the app still shows a number, but it is the wrong number.**

You can train wherever you like (a script, a notebook, Google Colab…). Training in the same
environment where you run the app (step 2 below) avoids version problems.

---

## Run it on your own computer

You need **Python 3.10–3.13** (3.12 is a safe choice).

**1. Get the code.** Click the green **Code** button on GitHub → **Download ZIP**, and unzip it.
Open a terminal in that folder.

**2. Install the packages** in a virtual environment (a private set of packages for this project):

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This downloads TensorFlow, which is big, so it can take a few minutes.

**3. (Optional) Rebuild the database.** `ehr.db` is already there. If you change `database.sql`, run:

```bash
python setup_database.py
```

**4. Add your model.** Train it (see [Your task](#your-task-train-the-model)) and put `model.keras` in this folder.

**5. Check that the model works on its own** (before involving the web app):

```bash
python prediction.py
```

You should see the model's input for Susan Foreman and a prediction between 0 and 1.

**6. Start the app:**

```bash
streamlit run app.py
```

Your browser opens the app at http://localhost:8501. Stop it with `Ctrl+C` in the terminal.

---

## Put it online (free)

You can host the app on [Streamlit Community Cloud](https://share.streamlit.io).

1. Put the files in your own GitHub repository, **including your `model.keras`**.
   (On GitHub: **Add file → Upload files**, then drag the files in.)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → **Yup, I have an app**.
4. Choose your repository, branch `main`, and main file `app.py`.
5. Click **Advanced settings**, choose **Python 3.12**, and click **Save**.

The first start takes a few minutes, because TensorFlow has to be installed.
Apps that nobody visits for a while go to sleep; the next visitor wakes them up.

---

## When something goes wrong

| What you see | What to do |
|---|---|
| `model.keras not found` | Save your model as `model.keras` in the same folder as `app.py` (and upload it, if the app is online). |
| `model.keras exists but could not be loaded` | Save it with `model.save("model.keras")`. If it works on your computer but not online, see the last row. |
| `Your model does not fit this app yet` | Read the message: it tells you which requirement (number of inputs, one output, values between 0 and 1) is not met. |
| `ehr.db not found` or "no patients" | Run `python setup_database.py` and check `database.sql`. |
| `ModuleNotFoundError: No module named 'streamlit'` (or `keras`) | Activate the virtual environment and run `pip install -r requirements.txt`. |
| The model works on your computer but not online | Your computer and the cloud have different TensorFlow versions. Run `python -c "import tensorflow as tf; print(tf.__version__)"` where you trained the model, and pin that version in `requirements.txt`, e.g. `tensorflow==2.21.0`. |
| Where are the error messages online? | Open your app and click **Manage app** (bottom right) to see the logs. |

---

## Going further: a different model or dataset

Want to use another dataset or more inputs? These are the places that have to agree with each other.
Finding out exactly *what* to change in each one is part of the exercise.

- **Your training code**: the data, the model, and the file it is saved to.
- **`database.sql`**: the columns and patient values your model needs.
- **`FEATURES` in `prediction.py`**: the model's inputs, **in the same order as during training**.
- **`app.py`**: which value(s) the user can type in.
- **`requirements.txt`**: any extra packages your model needs.
- **Preprocessing**: whatever you did to the inputs during training, the app must do in exactly the same way.

## Things to think about

- How did you measure your model's accuracy, and on which data? Why is accuracy on the training data too optimistic?
  How accurate would a "model" be that always says "no diabetes"? (About 65% of patients in the data don't have diabetes.)
- The five patients in `ehr.db` are copied from the Pima data. If your model was trained on them, what does that mean for the predictions you see?
- In this dataset a value of 0 often means "not measured": almost half the patients have insulin = 0.
  Susan Foreman's insulin is 0. Did you do anything about this when training? What does your model think that 0 means?
- The prediction is a number between 0 and 1, not a yes/no. Where would you put the cut-off, and who should decide?
  Does 0.70 really mean "a 70% chance"?
- The form accepts blood pressures up to 400, but the highest diastolic blood pressure in the data is 122.
  Try some extreme values. What happens, and should the app warn the user?
- The data come from women aged 21 and older of Pima heritage
  ([Smith et al., 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)). Would you use this model for other patients?

---

## Credits and license

Based on [ace-dvm/Python_ML_web_app_demo](https://github.com/ace-dvm/Python_ML_web_app_demo).
The data are the Pima Indians Diabetes dataset
([Smith et al., 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)).

Like the original, this code is licensed under the GNU General Public License v3.0 (see `LICENSE`).
