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

 ⚙️ Workflow

 1️⃣ Data collection, Preprocessing and Feature Engineering
Cheminformatics dataset containing molecular structures (used in SMILES format) with some associated properties.

<img width="948" height="377" alt="image" src="https://github.com/user-attachments/assets/e093c862-cd68-4497-80a2-4ab5f9167e5c" />
Among the compounds analyzed, Fenfuram, Citral, and Thiophene demonstrated balanced physicochemical properties and favourable drug-likeness, making them promising scaffolds for further development, whereas Amygdalin and Picene exhibited significant limitations in solubility, permeability, or structural flexibility.

<img width="948" height="291" alt="image" src="https://github.com/user-attachments/assets/4e20d228-dcd0-405c-bf54-57e3ef47ad2f" />
Amygdalin was replaced with Benzothiazole due to poor pharmacokinetics, resulting in a final set—Fenfuram, Citral, Picene, Thiophene, and Benzothiazole—whose molecular descriptors (MW, logP, solubility, HBD/HBA, PSA, rotatable bonds) indicated that Citral, Benzothiazole, Thiophene, and Fenfuram had favorable drug-likeness, while Picene showed poor bioavailability

Molecular Fingerprint Generation and Final Dataset Optimization:

<img width="948" height="272" alt="image" src="https://github.com/user-attachments/assets/5b5c48f4-ef95-40ed-a5d1-79f03b8117dd" />
RDKit-generated Morgan fingerprints (radius 2, 2048 bits) were obtained for five optimized compounds—Fenfuram, Citral, Thiophene, Benzothiazole, and Estradiol (replacing poorly drug-like Picene)—providing a robust basis for similarity screening and QSAR modeling.             


- SMILES canonicalization
- Duplicate and invalid molecule removal
- Z-score normalization
- 80/20 train-test split (random_state=42)


## 📊 Machine Learning Models

Although the dataset size was limited to 100 compounds, appropriate measures including feature selection, normalization, and cross-validation were employed to minimize overfitting and ensure model robustness. Regularization strategies and performance evaluation on independent test data were also used to validate the predictive capability of the models."

 🔹 Random Forest Regressor (IC₅₀ Prediction)

<img width="536" height="545" alt="image" src="https://github.com/user-attachments/assets/6aa6de7d-e340-4ab6-8372-420d61dadd9e" />

<img width="648" height="451" alt="image" src="https://github.com/user-attachments/assets/7b52d4c9-3e3f-4f55-a11b-a0ee4d08e330" />

The Random Forest Regressor demonstrated consistent predictive performance. The predicted versus actual plot  shows that data points are distributed around the identity line. The model achieved a test R² of approximately 0.85–0.87, with a training R² of 0.947.
Residual analysis indicated that residuals were centered around zero across the prediction range, with values approximately between −0.6 and 1.0. No apparent systematic pattern was observed in the residual distribution.

- 5-fold GridSearchCV (540 model fits)
- Hyperparameter tuning:
  - n_estimators = 300
  - max_depth = 10
  - min_samples_split = 2

Residual analysis confirmed strong generalization and minimal overfitting.

Feature Importance plot 

<img width="632" height="433" alt="image" src="https://github.com/user-attachments/assets/91aa0cab-bb0a-487d-9470-0bcf85b01248" />

SHAP analysis showed MolLogP (~0.75) and MolWt (~0.15) as key predictors, while other descriptors (e.g., rotatable bonds, TPSA, H-bond counts) had minimal impact (<0.1), indicating solubility and size chiefly drive model predictions.

 🔹 HistGradientBoosting Regressor
      
   <img width="581" height="522" alt="image" src="https://github.com/user-attachments/assets/3e612538-f3d8-41fc-9d6c-77dff7f05042" />

The HistGradientBoostingRegressor achieved R² = 0.753 on the test set, explaining ~75% of variance in biological activity. Predicted vs. actual values aligned with the identity line but showed moderate errors, with larger deviations at higher values, indicating reduced prediction accuracy in this range.

<img width="636" height="433" alt="image" src="https://github.com/user-attachments/assets/2cc247b6-9182-4bbe-a691-96b03a59bde9" />
Residual analysis revealed residuals ranging from -0.75 to 1.25, centered around zero with no systematic pattern across predicted values (2.0 to 5.0), supporting no major prediction bias

---
 🔹 Random Forest Classifier (Toxicity Prediction)

Distribution of Toxicity Classes in the Dataset
<img width="711" height="156" alt="image" src="https://github.com/user-attachments/assets/11ede4ac-a994-4d41-b1c3-cc924bd16801" />

The curated dataset comprised 100 chemical compounds with validated toxicity profiles, exhibiting a moderate class imbalance with 56 toxic compounds (56%) and 44 non-toxic compounds (44%) as detailed in Toxicity data Table .

<img width="732" height="220" alt="image" src="https://github.com/user-attachments/assets/c1dff498-e9f9-43d9-9e8a-9da9de8e4462" />

To address class imbalance in the training set, the Synthetic Minority Oversampling Technique (SMOTE) was applied. Prior to balancing, the training set contained 45 active and 35 inactive samples. Following SMOTE application, both classes were equally represented, with 45 samples each, resulting in a balanced training distribution.

<img width="852" height="351" alt="image" src="https://github.com/user-attachments/assets/fe07578f-5886-4fdd-8686-8376ed495d93" />

The Random Forest classifier demonstrated robust predictive performance with an overall accuracy of 85% and exceptional discriminative ability. Detailed performance metrics revealed class-specific variations, with perfect recall for toxic compounds and conservative prediction tendencies as presented in random forest classifier metrics table.

<img width="788" height="469" alt="image" src="https://github.com/user-attachments/assets/476c5d9d-2f67-4d5d-a603-b1e130a8ba3b" />

<img width="794" height="551" alt="image" src="https://github.com/user-attachments/assets/093cd2a9-bc5e-458a-a003-58e70b8fcab2" />

Feature importance analysis revealed that Molecular Weight (28.0%), Log P (23.0%), and IC-50 (20.0%) were the most influential predictors of toxicity in the Random Forest model, collectively accounting for 71% of the predictive power, followed by Polar Surface Area (13.0%), while structural descriptors such as Number of Rotatable Bonds, Number of Rings, Number of H-Bond Donors, and Minimum Degree contributed minimally (7.0%, 5.0%, 4.0%, and 2.0%, respectively). 

Confusion Matrix Analysis and ROC Performance

<img width="670" height="492" alt="image" src="https://github.com/user-attachments/assets/c72d9908-83fa-48c7-8254-fdc431749bde" />

<img width="775" height="663" alt="image" src="https://github.com/user-attachments/assets/45ba69b1-261b-4e01-8a93-b4078bb87fc6" />

The Random Forest Classifier achieved excellent predictive performance. On the independent test set, the model yielded an accuracy of 95%, with a sensitivity of 100% and specificity of 89% as shown in Metrics table. The confusion matrix (Fig) shows 11 true positives, 8 true negatives, 1 false positive, and 0 false negatives, with an AUC-ROC score of 0.9596. These results suggest strong predictive capacity with optimal safety profile, as no toxic compounds were missed. However, the relatively small test set (n = 20) and the single false positive warrant validation on larger, independent datasets to confirm generalizability across diverse chemical spaces.

 🔹 Feedforward Neural Network (LogP Regression)



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

