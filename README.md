# Disease Classification — Multi‑class & Binary (Acute vs Chronic)

Final project for the **Supervised Learning** course (BSc Data Science & Artificial Intelligence, SUPSI — A.Y. 2025/2026).

Starting from ~20,000 patient medical records (age, gender, symptoms, systolic blood pressure, stomach acidity, …) the project tackles two tasks:

1. **Multi‑class classification** — predict the disease among **30 classes**
2. **Binary classification** — predict whether the disease is **acute or chronic** (via `disease_mapping.json`)

## Pipeline

- **EDA**: distributions, correlation matrix, box plots, class balance, realism checks against external references
- **Preprocessing**: removal of index/ID columns, symptoms string → multi‑hot binary columns (`MultiLabelBinarizer`), gender encoding, label‑encoded target, standard scaling
- **Hyperparameter tuning** with `GridSearchCV` and cross‑validation for every model
- **Stacking** with a proper 60 / 20 / 20 split (base learners / blender / final test)
- **Model persistence** with `joblib` (`load_or_train_model` loads a saved model or retrains it)

## Results — Multi‑class (30 diseases)

| Model | Best hyperparameters | Test accuracy | Train–test gap |
|---|---|---|---|
| Softmax Regression (ℓ2) | C = 125 | 45.74 % | 5.08 % |
| Decision Tree | entropy, max_depth = 10, min_samples_leaf = 5 | 82.18 % | 0.91 % |
| Random Forest | 200 trees, max_features = sqrt, min_samples_split = 10 | 82.85 % | 6.38 % |
| **Gradient Boosting** | lr = 0.01, 50 trees, max_depth = 3 | **82.87 %** | **0.11 %** |
| Stacking (LogReg blender) | Softmax + DT + RF + GB | 82.87 % | – |
| MLP (Adam, ReLU) | (50, 50), lr = 0.01, batch = 64 | 80.32 % | 0.62 % |

Random guessing on 30 classes would give ≈ 3.3 %. Systolic blood pressure is the most discriminative feature (≈ 30 % importance in the Random Forest).

## Results — Binary (Acute vs Chronic)

| Model | Test accuracy | Recall Acute | Recall Chronic | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 63.19 % | 0.19 | 0.89 | 0.64 |
| **Random Forest** | **95.54 %** | **0.93** | **0.97** | **0.96** |
| MLP | 91.00 % | 0.89 | 0.92 | 0.95 |

Non‑linear models are essential here: the linear model collapses toward the majority class (Chronic, 63 %), while Random Forest and MLP separate both classes well.

## Repository contents

| Path | Description |
|---|---|
| `Loi_Mattia_finalproject_SL.ipynb` | Full notebook (EDA, preprocessing, all models, evaluation) |
| `Loi_Mattia_finalproject_SL.html` | HTML export of the notebook (viewable without Jupyter) |
| `Loi_Mattia_finalproject_report.pdf` | Written report with discussion of the results |
| `Dataset_FinalProject_AA2526.csv` | Dataset (20,167 samples, 9 features) |
| `disease_mapping.json` | Disease → acute / chronic mapping |
| `saved_models/` | Trained models (`.joblib`) |

## How to run

```bash
git clone <this-repo-url>
pip install -r requirements.txt
jupyter notebook Loi_Mattia_finalproject_SL.ipynb
```

The notebook loads the models from `saved_models/` when available; any missing model is retrained automatically and saved there.

## Tech stack

Python · NumPy · pandas · scikit‑learn · joblib · Matplotlib · Seaborn · missingno

## Author

**Mattia Loi** — BSc Data Science & Artificial Intelligence, SUPSI
