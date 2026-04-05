# 🧬 AI-Driven QSAR & Virtual Screening Pipeline for Anti-Cancer Drug Discovery

## 📌 Overview

This project presents an end-to-end AI-driven computational drug discovery pipeline integrating:

- Cheminformatics (RDKit)
- Machine Learning (Scikit-learn)
- Deep Learning (TensorFlow)
- Model Interpretability (SHAP)
- Molecular Docking (AutoDock Vina)

The objective is to predict biological activity (IC₅₀), toxicity, and drug-likeness of small molecules and integrate docking scores to prioritize anti-cancer scaffolds targeting Caspase-9, BCL-2, and VEGF.

This repository demonstrates a reproducible and interpretable AI workflow for early-stage drug discovery.

---

## 🧠 Project Highlights

- 2048-bit Morgan fingerprint generation (radius = 2)
- Random Forest regression with strong generalization (R² = 0.929 test)
- Toxicity classification with AUC-ROC = 0.959
- SMOTE-based class balancing
- SHAP-based model interpretability
- Feedforward Neural Network regression (MAE ≈ 0.83)
- Molecular docking integration with ML consensus scoring

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing
- SMILES canonicalization
- Duplicate and invalid molecule removal
- Z-score normalization
- 80/20 train-test split (random_state=42)

### 2️⃣ Descriptor Engineering
Computed using RDKit:
- Molecular Weight (MW)
- LogP
- TPSA
- H-bond donors/acceptors
- Rotatable bonds
- FractionCSP3
- Ring count
- Morgan fingerprints (2048 bits)

---

## 📊 Machine Learning Models

Although the dataset size was limited to 100 compounds, appropriate measures including feature selection, normalization, and cross-validation were employed to minimize overfitting and ensure model robustness. Regularization strategies and performance evaluation on independent test data were also used to validate the predictive capability of the models."

 🔹 Random Forest Regressor (IC₅₀ Prediction)
Predictive accuracy plot
<img width="536" height="545" alt="image" src="https://github.com/user-attachments/assets/6aa6de7d-e340-4ab6-8372-420d61dadd9e" />
Residual Analysis plot
<img width="648" height="451" alt="image" src="https://github.com/user-attachments/assets/737be947-e524-4222-b563-4198107ea625" />
The Random Forest Regressor demonstrated consistent predictive performance. The predicted versus actual plot  shows that data points are distributed around the identity line. The model achieved a test R² of approximately 0.85–0.87, with a training R² of 0.947.
Residual analysis indicated that residuals were centered around zero across the prediction range, with values approximately between −0.6 and 1.0. No apparent systematic pattern was observed in the residual distribution.

- 5-fold GridSearchCV (540 model fits)
- Hyperparameter tuning:
  - n_estimators = 300
  - max_depth = 10
  - min_samples_split = 2

Residual analysis confirmed strong generalization and minimal overfitting.
Feature Importance plot 
<img width="632" height="433" alt="image" src="https://github.com/user-attachments/assets/d7a886de-c7f7-4af6-b639-a9c6cf59aa18" />
SHAP analysis showed MolLogP (~0.75) and MolWt (~0.15) as key predictors, while other descriptors (e.g., rotatable bonds, TPSA, H-bond counts) had minimal impact (<0.1), indicating solubility and size chiefly drive model predictions.
---

### 🔹 HistGradientBoosting Regressor

- Test R² = 0.753
- Slight overfitting observed
- Demonstrates comparative model benchmarking

---

### 🔹 Random Forest Classifier (Toxicity Prediction)

- Accuracy = 95%
- AUC-ROC = **0.959**
- Sensitivity = 100%
- SMOTE applied to training set only
- Stratified k-fold cross-validation (mean accuracy = 0.99)

---

### 🔹 Feedforward Neural Network (LogP Regression)

Architecture:
```
Input (8 features)
→ Dense (64)
→ Dense (32)
→ Output (1)
```

- Test MAE ≈ 0.83
- Convergence by ~30 epochs
- Moderate overfitting observed

---

## 🔍 Model Interpretability (SHAP)

SHAP analysis identified:

- MolLogP as dominant predictor
- Molecular Weight as secondary contributor

This aligns with medicinal chemistry principles linking lipophilicity and size to bioactivity.

---

## 🧪 Molecular Docking Integration

Due to computational resource limitations and project duration constraints, molecular docking simulations were conducted on five representative compounds selected from the active and non-toxic subset.

Docking performed using AutoDock Vina.

Targets:
- Caspase-9
- BCL-2
- VEGF

Consensus Filtering Criteria:
- Docking score < -7.0 kcal/mol
- High ML-predicted activity
- Favorable ADMET properties

Promising scaffolds identified:
- Benzothiazole
- Citral
- Thiophene (optimization required)

---

## 📂 Repository Structure

```
ai-qsar-drug-discovery/
│
├── data/
├── notebooks/
├── src/
├── results/
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

- Python 3.x
- RDKit
- Scikit-learn
- TensorFlow
- SHAP
- NumPy / Pandas
- Matplotlib / Seaborn
- AutoDock Vina

---

## 🚀 Key Technical Contributions

- End-to-end reproducible ML pipeline
- Hyperparameter optimization with cross-validation
- SMOTE class imbalance handling
- SHAP explainability integration
- ML + docking hybrid consensus modeling
- Strong generalization with validated performance metrics

---

## 📈 Future Improvements

- Expansion to larger datasets (>10k compounds)
- Graph Neural Networks (GNNs)
- DeepChem integration
- External validation datasets
- Deployment as API or web application

---

## 👩‍🔬 Author

Dr. J. Srimathi Devi, PhD  
Computational Biologist | AI in Drug Discovery  
Open to global AI Startups & Pharma platforms & computational biology roles  

GitHub: https://github.com/srimathi-del/ai-qsar-drug-discovery/edit/main/README.md
LinkedIn: https://linkedin.com/in/www.linkedin.com/in/dr-j-srimathi-devi-5b335158
---

## 📜 License

This project is licensed under the MIT License.

