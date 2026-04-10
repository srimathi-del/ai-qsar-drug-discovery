# 🧬 AI-Driven QSAR & Virtual Screening Pipeline for Anti-Cancer Drug Discovery

## 📌 Overview

This project presents an end-to-end AI-driven computational drug discovery pipeline integration outlined in Graphical Abstract

 <img width="948" height="635" alt="image" src="https://github.com/user-attachments/assets/44fa6d5b-10e2-4dd1-a450-22a0ce865534" />

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

Sequential Model Architecture and Training Dynamics
<img width="896" height="486" alt="image" src="https://github.com/user-attachments/assets/f0e0e081-2efe-437e-80f3-3b9b04182804" />

"Training and Validation Loss Over 40 Epochs"

<img width="709" height="523" alt="image" src="https://github.com/user-attachments/assets/f54f57aa-86d4-437f-9e57-1ff39b7f8a31" />

The Feedforward Neural Network (FNN) with three fully connected layers (8→64→32→1; 2,689 parameters) was trained on 100 synthetic samples (8 features, 1 target) for 40 epochs. Training loss (MSE: 10.91→0.42; MAE: 2.82→0.50) and validation loss (MSE: 15.77→0.90; MAE: 3.35→0.79) showed strong learning and moderate generalization, with convergence by ~25–30 epochs (p < 0.05).

Test Performance and Predictive Output
<img width="876" height="132" alt="image" src="https://github.com/user-attachments/assets/d3f268fb-7d92-4b9e-9723-55a2533abc7d" />

Evaluation on the independent test dataset yielded a loss of 1.45 (MSE) and an MAE of 0.83 . These values, although slightly higher than the final validation loss (0.90) and MAE (0.79), remain consistent with the overall trend and indicate satisfactory generalization capability of the model.

"Single-Step Model Prediction Output"

<img width="673" height="545" alt="image" src="https://github.com/user-attachments/assets/0a9b2064-772d-4a72-b90b-a59554db0ed6" />

The regression model showed good predictive performance on the test dataset (n = 20), with predicted values ranging from 0.27 to 5.64. The model achieved a Mean Absolute Error (MAE) of 0.83, indicating that the predictions were on average less than one unit away from the true values. Stable training and validation performance suggested that the model learned meaningful patterns without overfitting and was able to generalize well to unseen data.
---

## 🧪 Molecular Docking Integration

Due to computational resource limitations and project duration constraints, molecular docking simulations were conducted on five representative compounds selected from the active and non-toxic subset.

In Silico Docking of Citral to Caspase-9 for Binding Affinity Prediction

"Auto Dock Vina Docking Scores for Citral with Caspase-9

<img width="840" height="415" alt="image" src="https://github.com/user-attachments/assets/bb20a8b3-702c-40b2-86b2-d91b6ce15ee1" />

Molecular docking of Citral with Caspase-9 produced nine binding modes, with affinities ranging from –4.06 kcal/mol (Mode 1, most favourable) to +5.10 kcal/mol (Mode 9, least favourable). The optimal pose (Mode 1) served as the reference for RMSD calculations, while Modes 2–3 showed similar stable orientations (RMSD 2.1–2.9 Å). Modes 4–7 had moderate affinities (–3.57 to –3.46 kcal/mol), Mode 8 suggested an alternative orientation, and Mode 9 displayed an unfavourable positive affinity with higher RMSD values .

Molecular Docking of Citral with Caspase-9 Active Site.

<img width="997" height="522" alt="image" src="https://github.com/user-attachments/assets/31129400-1319-4582-829f-c6f1fca855ab" />

Structural analysis of the top-ranked pose revealed that Citral occupies a hydrophobic pocket within Caspase-9’s active site, stabilized by van der Waals and weak polar interactions. The ligand established hydrophobic contacts with LEU142, LEU149, ALA141, and GLY147, while polar interactions involved ARG143, ASN148, SER144, and GLN373. Importantly, Citral was positioned in proximity to the catalytic cysteine residue (CYS287), suggesting potential implications in modulation of Caspase-9 activity. Additional stabilizing contacts with TRP374, PHE371, and ASP369 further supported the binding orientation. Intermolecular distances ranged between 2.9–4.6 Å, with shorter contacts representing van der Waals and weak polar interactions, while longer contacts indicated hydrophobic stabilization.

In Silico Docking of Benzothiazole to Veg-F for Binding Affinity Prediction

<img width="802" height="492" alt="image" src="https://github.com/user-attachments/assets/5d363b40-97e9-4f95-8abf-04c387cca7de" />

Auto Dock Vina generated nine benzothiazole–VEGF binding poses with affinities from –4.51 to –3.70 kcal/mol. The best pose (–4.51 kcal/mol) and five others (Modes 1–6) clustered closely, indicating a stable primary site, while Modes 7–9 showed weaker secondary binding.

"Binding Modes and Interaction Profile of Thiazole with VEGF"

<img width="989" height="492" alt="image" src="https://github.com/user-attachments/assets/46f5c223-3d95-4a20-89be-9ac8c8989a58" />

RMSD analysis identified three clusters: Cluster I (Modes 1, 3–6) in the primary pocket (1.3–3.1 Å), Cluster II (Modes 2, 8) at alternative sites (~14–17 Å), and Cluster III (Modes 7, 9) as transitional (3.9–4.3 Å). The optimal VEGF pose (Fig. 7) showed three interaction zones: π–π stacking with PHE-47, TYR-45, TYR-25; polar contacts with HIS-90, ASP-61, GLU-42/64, ASN-34, SER-50; and hydrophobic encapsulation by ILE-30, VAL-52, LEU-48, MET-81, with the thiazole nitrogen and benzene ring enhancing amphiphilic complementarity within the 400–500 Å³ pocket.

In Silico Docking of Thiophene to Bcl-2 for Binding Affinity Prediction

Auto Dock Vina Docking Results for Thiophene–BCL2 Interaction

<img width="778" height="437" alt="image" src="https://github.com/user-attachments/assets/eb66d07d-1cb6-407a-b26b-8801be16094a" />

Auto Dock Vina docking of thiophene with BCL-2 yielded nine poses (–3.009 to –2.024 kcal/mol); the best pose (–3.009 kcal/mol) and Modes 2–3 were structurally similar, while Modes 7–9 showed higher RMSD and weaker, alternative orientations.

Binding Site Analysis of Thiophene with Bcl-2 Active Site.

<img width="739" height="492" alt="image" src="https://github.com/user-attachments/assets/8d1d7ee5-bdc1-4651-b82e-ca900e30810c" />

The optimal thiophene binding pose (Mode 1) localized within BCL-2’s BH3-binding groove and was stabilized by multiple non-covalent interactions. Strong hydrogen bonds formed with ARG127 and HIS184 (~2.3 Å), while hydrophobic residues PHE153, VAL132, LEU134, and VAL135 created a lipophilic pocket enhancing van der Waals interactions. Aromatic residues PHE153 and TYR18 allowed potential π–π stacking with the thiophene ring, and additional polar contacts with GLN190 and HIS184 further contributed to electrostatic stabilization .

--

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

