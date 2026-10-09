# A toy clinical decision support app, built with Streamlit

This repo contains a small demo for students: a web app that lets a user

1. pick a patient from a tiny (fictional) health record,
2. optionally type in a different blood pressure value, and
3. see what a machine-learning model predicts about that patient's chance of diabetes.

The app is ready to go, except for one thing: **the model. Training it is your job.**
After that, you will use this repo as the starting point for your own app to diagnose cancer
(see [Part 2](#part-2-build-your-own-app-to-diagnose-cancer)).

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

### Why is there a database?

Strictly speaking, an app that only takes values from the user and returns a prediction doesn't
need a database. So why is there one?

The database plays the role of the hospital's **electronic health record (EHR)**. In a real clinical
decision support system, the clinician doesn't type in the patient's data: lab results, vital signs
and history are already in the record. The system reads them from there, and the clinician only
selects a patient and, at most, corrects or adds a value. That is what this app does: it reads the
patient from `ehr.db`, and the user can change the blood pressure for the prediction
(nothing is written back to the database).

This matters for three reasons:

- **Decision support works best when it fits into the clinician's workflow.** We don't want clinicians to re-type data that is already saved.
- **Typing is error-prone.** 
- **The data and the model are designed separately.** The database has its own column names,
  units and missing values. Getting from "database column" to "model input, in the right order and
  with the right preprocessing" is a real integration problem, and you will meet it in Part 2.

`ehr.db` is a very small stand-in. A real EHR is a separate system that applications reach through
a standard interface.

---

## Your task: train the model

Train a neural network that predicts diabetes, using the **Pima Indians Diabetes** dataset:
[pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
(768 patients, no header row; the 9th column is the outcome: 1 = diabetes, 0 = no diabetes).

You decide how to build and train it. To work in this app, your model must:

1. **Take 8 inputs, in this order:**
   pregnancies, glucose, diastolic blood pressure, skin thickness, insulin, BMI,
   diabetes pedigree function, age. This is the column order of the CSV, and the order
   of `FEATURES` in `prediction.py`.
2. **Accept the raw values** as they are in the database (e.g. glucose = 148, age = 50).
   If your model needs scaled inputs, either put the scaling *inside* the model (use a Keras
   `Normalization` layer, not a `Lambda` layer, which won't load), or do it in the
   `preprocess` function in `prediction.py`.
3. **Output one number between 0 and 1** for each patient: the model's estimate that the patient has diabetes.
4. **Be a Keras model saved as `model.keras`**, with `model.save("model.keras")`,
   and placed in the same folder as `app.py`. Don't rename a file saved in another format
   (such as `model.h5`): it won't load.

The app checks points 1 and 3 when it starts and tells you if something doesn't fit.
It cannot check point 2 or the order in point 1, so double-check those yourself:
**if the order or scaling is wrong, the app still shows a number, but it is the wrong number.**

You can train wherever you like (a script, a notebook, Google Colab…). Training in the same
environment where you run the app (step 2 below) avoids version problems.

---

## Run it on your own computer

You need **Python 3.10–3.13** (3.12 is a safe choice). TensorFlow does not support newer
versions yet, so check with `python --version` first.

> **`python`, `python3` or `py`?** The commands below use `python`. On macOS and Linux
> you may need `python3` instead; on Windows, `py`. Once the virtual environment is
> active (step 2), plain `python` works everywhere.

If something doesn't work, look at [Troubleshooting](#troubleshooting) at the end of this README.

**1. Get the code.** Click the green **Code** button on GitHub → **Download ZIP**, and unzip it.
Open a terminal in that folder.

**2. Install the packages** in a virtual environment (a private set of packages for this project).

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows (Command Prompt or PowerShell):

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

This downloads TensorFlow, which is big, so it can take a few minutes.
You need to activate the environment again (the second line) every time you open a new terminal.

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

Two things that look like problems but aren't:
- The very first time, Streamlit asks for your email address in the terminal. Just press Enter.
- TensorFlow prints many messages when it starts (about CUDA, GPUs or oneDNN).
  On a normal laptop these are harmless.

---

## Put it online (free)

You can host the app on [Streamlit Community Cloud](https://share.streamlit.io).

1. Put the files in your own GitHub repository, **including your `model.keras`**.
   (On GitHub: **Add file → Upload files**, then drag in the files *inside* the folder,
   not the folder itself, so that `app.py` sits at the top level of the repository.)
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

## Part 2: build your own app to diagnose cancer

Once the Pima app works and you understand what every file does, you will turn it into
**your own decision support app that helps diagnose a type of cancer**, using the cancer dataset
for your assignment. Start from a copy of this repo (your own new GitHub repository) and change it step by step.

The app's structure stays the same: a database of (fictional) patients, a trained model, and a page
that shows the model's output for the selected patient. Almost everything *inside* that structure changes.

### What you will need to change

These are the places that have to agree with each other. Finding out exactly *what* to change
in each one is part of the assignment.

1. **Your model.** Explore your dataset first: which features are there, in which units, are values
   missing, and how many patients have cancer? Then train and evaluate your network on data it has
   not seen, and save it as `model.keras`. Decide which outcome is 1 (for example: 1 = malignant)
   and make sure the app says the same thing.
2. **`database.sql`**: a `patient` table whose columns are your features, and a few patient records.
   Use made-up patients or rows you held out from training, and never real data that could identify a person.
   Then run `python setup_database.py` again.
3. **`FEATURES` in `prediction.py`**: your feature names, the same as the database columns,
   **in the same order as during training**.
4. **Preprocessing**: whatever you did to the inputs during training (scaling, encoding, filling in
   missing values), the app must do in exactly the same way.
5. **`app.py`**: the title and texts on the page, and how the result is shown: a probability, a label, or both?
6. **Let the clinician edit any value.** In the Pima app the user can only change the blood pressure.
   In your app, the clinician must be able to change **any** of the patient's values before asking for a
   prediction. The values from the database are the starting point; changes are used for the prediction
   only and are not saved. This is a change to both `app.py` and `prediction.py` (look at `make_input`).
   Your page must also:
   - show clearly **which values were changed** compared to the record, so it is visible what the
     prediction was based on;
   - give every field a **sensible minimum and maximum**, based on your training data;
   - show the **record's values again** when the clinician switches to another patient.
7. **The checkpoint in `prediction.py`** (the part under `if __name__ == "__main__":`): example patients
   from your own data, so that `python prediction.py` still tests your model.
8. **`requirements.txt`**: any extra packages you use.
9. **`README.md`**: describe your own app, your data and your model.

**Tip:** change one thing at a time and run the app after each step. The error messages tell you
where it breaks. Test your app with a patient whose diagnosis you know.

---

## Credits and license

Based on [ace-dvm/Python_ML_web_app_demo](https://github.com/ace-dvm/Python_ML_web_app_demo).
The data are the Pima Indians Diabetes dataset
([Smith et al., 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)).

Like the original, this code is licensed under the GNU General Public License v3.0 (see `LICENSE`).
