# Space Debris Decay Risk: interactive walkthrough

A Streamlit app that runs the CSE 587 Phase 3 project end to end from a CSV:
cleaning rules C0 to C14, Bronze/Silver/Gold tables, a stratified train/test split,
and three models (Linear Regression, Logistic Regression, Random Forest).
Everything is trained when you open the app, so you can change the split and the forest settings
and watch the results change.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `space_debris_orbits_2026.csv` in the sidebar, or place it in `data/`.

## Free hosting, step by step

### Option 1: Streamlit Community Cloud (simplest)

1. Create a GitHub repository and push these files (not the CSV, see `data/README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, branch `main`, and main file `app.py`.
4. Click **Deploy**. The first start takes a few minutes while the packages install.
5. Open the app, then upload the CSV in the sidebar. Viewers can do the same.

### Option 2: Hugging Face Spaces

1. Create a new Space, choose SDK **Streamlit**, and upload these files.
2. The Space starts automatically. Upload the CSV in the sidebar.

### Option 3: Render (free web service)

1. Create a new Web Service from the repository.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit interface: Pipeline, Models, Predict, Explore and Notes tabs |
| `debris_core.py` | Cleaning rules, features, split, the three models, and single-object prediction |
| `requirements.txt` | Pinned versions that were tested |
| `data/` | Put the CSV here to load it automatically |

## Honest notes

- This is a pandas and scikit-learn mirror of the Spark notebooks. The Spark notebooks remain the reference run. The app shows the Spark numbers beside its own, and they can differ.
- The split uses a hash of `object_id` with a seed, the same idea as the notebook. The Random Forest is scikit-learn's, not Spark's.
- The Kepler period check runs here. The vis-viva speed check runs only in the Spark notebook.
- The data is probably synthetic. The predictions are for teaching and demonstration, not for real collision or re-entry decisions.
