# A toy clinical decision support app, built with Streamlit

This repo contains a small demo for students: a web app that lets a user

1. pick a patient from a tiny (fictional) health record,
2. optionally type in a different blood pressure value, and
3. see what a machine-learning model predicts about that patient's chance of diabetes.

It is a [Streamlit](https://streamlit.io) version of
[Python_ML_web_app_demo](https://github.com/ace-dvm/Python_ML_web_app_demo) (Flask + PythonAnywhere),
with the same data, the same model and the same five patients.

> ⚠️ **This is a technical demo. It must not be used to make clinical decisions.**

---

## What's in this repo

| File | What it does |
|---|---|
| `app.py` | The web app (the page you see in the browser). |
| `prediction.py` | Loads the model and turns a patient into a prediction. |
| `train_model.py` | Trains the model and saves it as `model.keras`. |
| `database.sql` | The patient records, written in SQL. |
| `setup_database.py` | Turns `database.sql` into the database file `ehr.db`. |
| `ehr.db` | The database file (already built for you). |
| `requirements.txt` | The Python packages the app needs. |

`model.keras` is **not** included: you create it in step 4 below.

## How the pieces fit together

```
 database.sql ──(setup_database.py)──▶ ehr.db ─────────┐
                                                       ▼
                                  app.py ◀──▶ prediction.py ◀── model.keras
                                    ▲                              ▲
                                    │                              │
                    you, in the browser          train_model.py (trains on the Pima data)
```

When you click something in the app, Streamlit re-runs `app.py` from top to bottom:
it reads the patients from `ehr.db`, builds the 8 numbers the model needs, asks the
model for a prediction and redraws the page.

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

**4. Train the model.** This takes about a minute and creates `model.keras`:

```bash
python train_model.py
```

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

1. Put the files in your own GitHub repository, **including `model.keras` and `ehr.db`**.
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
| `model.keras not found` | Run `python train_model.py` (and upload `model.keras` if the app is online). |
| `ehr.db not found` or "no patients" | Run `python setup_database.py` and check `database.sql`. |
| `ModuleNotFoundError: No module named 'streamlit'` (or `keras`) | Activate the virtual environment and run `pip install -r requirements.txt`. |
| `train_model.py` cannot download the data (e.g. a certificate error) | Download the CSV from the link in `train_model.py`, save it next to the script as `pima-indians-diabetes.data.csv`, and run it again. |
| The model loads on your computer but not online | Your computer and the cloud have different TensorFlow versions. Run `python -c "import tensorflow as tf; print(tf.__version__)"` and pin that version in `requirements.txt`, e.g. `tensorflow==2.21.0`. |
| Where are the error messages online? | Open your app and click **Manage app** (bottom right) to see the logs. |

---

## Make it your own

Want to use your own model or dataset? Here are the places that have to agree with each other.
Finding out exactly *what* to change in each one is part of the exercise.

- **The training script**: your data, your model, and the file it is saved to.
- **`database.sql`**: the columns and patient values your model needs.
- **`FEATURES` in `prediction.py`**: the model's inputs, **in the same order as during training**.
  Get the order wrong and the app still shows a number, but it is the wrong number.
- **`app.py`**: which value(s) the user can type in.
- **`requirements.txt`**: any extra packages your model needs (for example `scikit-learn`).
- **Preprocessing**: if your model was trained on scaled, encoded or imputed inputs, the app has to
  apply exactly the same steps before predicting.

## Things to think about

- `train_model.py` reports accuracy **on the data it was trained on**. Why is that too optimistic?
  How accurate would a "model" be that always says "no diabetes"? (About 65% of patients in the data don't have diabetes.)
- The five patients in `ehr.db` are copied from the training data. What does that mean for the predictions you see?
- In this dataset a value of 0 often means "not measured": almost half the patients have insulin = 0.
  Susan Foreman's insulin is 0. What does the model think that 0 means?
- The prediction is a number between 0 and 1, not a yes/no. Where would you put the cut-off, and who should decide?
  Does 0.70 really mean "a 70% chance"?
- The form accepts blood pressures up to 400, but the highest diastolic blood pressure in the data is 122.
  Try some extreme values. What happens, and should the app warn the user?
- The data come from women aged 21 and older of Pima heritage
  ([Smith et al., 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)). Would you use this model for other patients?

---

## Credits and license

Based on [ace-dvm/Python_ML_web_app_demo](https://github.com/ace-dvm/Python_ML_web_app_demo),
which follows this [Keras tutorial](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/).
The data are the Pima Indians Diabetes dataset
([Smith et al., 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)).

Like the original, this code is licensed under the GNU General Public License v3.0 (see `LICENSE`).
