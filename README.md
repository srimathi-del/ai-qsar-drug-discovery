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

### 🔹 Random Forest Regressor (IC₅₀ Prediction)

- Train R² = 0.925  
- Test R² = **0.929**
- 5-fold GridSearchCV (540 model fits)
- Hyperparameter tuning:
  - n_estimators = 100
  - max_depth = 10
  - min_samples_split = 10

Residual analysis confirmed strong generalization and minimal overfitting.

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
Open to global remote AI & computational biology roles  

GitHub: https://github.com/srimathi-del/ai-qsar-drug-discovery/edit/main/README.md
LinkedIn: https://linkedin.com/in/www.linkedin.com/in/dr-j-srimathi-devi-5b335158
---

## 📜 License

This project is licensed under the MIT License.

