#  Prédiction du Churn Client — Secteur Télécommunications

> Projet de Machine Learning supervisé pour prédire la résiliation d'abonnement client à partir du dataset IBM Telco Customer Churn.

---

##  Table des matières

- [Contexte]
- [Dataset]
- [Structure du projet]
- [Pipeline ML]
- [Modèles]
- [Résultats]
- [Installation]
- [Utilisation]
- [Technologies]
- [Insights Business]

---

##  Contexte

Le **churn** (attrition client) représente l'un des défis majeurs du secteur télécom : acquérir un nouveau client coûte 5 à 25 fois plus cher que d'en retenir un existant. 

Ce projet développe un système de prédiction permettant d'identifier les clients à risque **avant** leur départ, afin de déclencher des actions de rétention ciblées.

**Type de problème :** Classification binaire supervisée  
**Variable cible :** `Churn` — Yes (1) / No (0)  
**Application :** Département rétention client, scoring quotidien, campagnes marketing ciblées

---

##  Dataset

| Caractéristique | Valeur |
|----------------|--------|
| Source | IBM Sample Data — Telco Customer Churn |
| Taille | 7 043 clients × 21 variables |
| Variables numériques | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Variables catégorielles | 17 (contrat, services, paiement...) |
| Déséquilibre | 73.5% Non-Churn / 26.5% Churn |

### Variables clés
- **tenure** — Ancienneté du client (mois)
- **Contract** — Type de contrat (mensuel, annuel, 2 ans)
- **MonthlyCharges** — Charges mensuelles
- **InternetService** — Type de service internet
- **PaymentMethod** — Méthode de paiement
- **OnlineSecurity / TechSupport** — Services additionnels

---

##  Structure du projet

```
churn-prediction/
│
├── Projet_final_Validation.ipynb    # Notebook principal complet
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── outputs/
│   ├── best_churn_model.pkl         # Meilleur modèle sauvegardé
│   ├── scaler.pkl                   # StandardScaler fitted
│   └── figures/
│       ├── 01_distribution_churn.png
│       ├── 02_boxplots_outliers.png
│       ├── 03_distributions_numeriques.png
│       ├── 04_churn_par_categorie.png
│       ├── 05_correlation_matrix.png
│       ├── 06_smote_comparison.png
│       ├── 07_cv_results_comparison.png
│       ├── 08_optimization_comparison.png
│       ├── 09_regularization_effect.png
│       ├── 10_final_metrics_comparison.png
│       ├── 11_confusion_matrices.png
│       ├── 12_roc_curves.png
│       ├── 13_precision_recall_curves.png
│       ├── 14_learning_curves.png
│       ├── 15_overfitting_analysis.png
│       └── 16_feature_importance.png
│
└── README.md
```

---

##  Pipeline ML

```
Données brutes (7043 × 21)
        │
        ▼
┌─────────────────────────────┐
│   1. Analyse exploratoire   │  Distributions, corrélations,
│      (EDA)                  │  déséquilibre, outliers
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   2. Prétraitement          │  Imputation TotalCharges
│                             │  Label Encoding (binaires)
│                             │  One-Hot Encoding (multi-classes)
│                             │  Suppression customerID
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   3. Train/Test Split       │  80% / 20% — Stratifié
│      + Standardisation      │  StandardScaler (fit sur train)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   4. SMOTE                  │  Équilibrage classe minoritaire
│                             │  (Churn: ~1400 → ~4200)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   5. Validation Croisée     │  Stratified K-Fold (k=5)
│      (5 modèles)            │  Accuracy, F1, Recall, AUC
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   6. Optimisation           │  RandomizedSearchCV (n_iter=30)
│      Hyperparamètres        │  Scoring: AUC-ROC
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   7. Évaluation finale      │  Test set (jamais vu)
│      + Analyse overfitting  │  ROC, Precision-Recall, CM
└─────────────────────────────┘
```

---

##  Modèles

Cinq modèles ont été comparés, couvrant différentes familles algorithmiques :

| Modèle | Famille | Avantage principal |
|--------|---------|-------------------|
| **Régression Logistique** | Linéaire | Interprétabilité maximale |
| **Random Forest** | Ensemble (Bagging) | Robuste, importance des features |
| **XGBoost** | Ensemble (Boosting) | État de l'art tabulaire |
| **SVM** | Noyau | Efficace en haute dimension |
| **KNN** | Instance-based | Aucune hypothèse de distribution |

### Régularisation appliquée
- **Régression Logistique** : L1 / L2 via paramètre `C`
- **Random Forest** : `max_depth`, `min_samples_split`, `min_samples_leaf`
- **XGBoost** : `learning_rate`, `max_depth`, `subsample`, `scale_pos_weight`
- **SVM** : paramètre `C` et `gamma`

---

##  Résultats

### Validation croisée (5-Fold Stratifié)

| Modèle | Accuracy | F1-Score | Recall | AUC-ROC |
|--------|----------|----------|--------|---------|
| XGBoost | ~0.80 | ~0.57 | ~0.54 | **~0.85** |
| Random Forest | ~0.80 | ~0.55 | ~0.52 | ~0.84 |
| Régression Logistique | ~0.77 | ~0.54 | ~0.56 | ~0.83 |
| SVM | ~0.78 | ~0.53 | ~0.51 | ~0.83 |
| KNN | ~0.75 | ~0.49 | ~0.48 | ~0.79 |

> Les valeurs exactes dépendent de l'exécution — se référer aux outputs du notebook.

### Analyse du surapprentissage (Gap Train-Test)

- Gap < 0.05 pour tous les modèles → **bonne généralisation**
- Modèles stables confirmés par les learning curves

### Modèle recommandé

**XGBoost** — meilleur AUC-ROC global  
**Régression Logistique** — recommandée pour l'explication aux parties prenantes

---

## Installation

### Prérequis
- Python 3.13+
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/<ton-username>/churn-prediction.git
cd churn-prediction

# 2. Créer un environnement virtuel 
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
```

---

## Utilisation

### Exécuter le notebook complet
```bash
jupyter notebook Projet_final_Validation.ipynb
```

### Charger le modèle sauvegardé pour prédire
```python
import joblib
import pandas as pd

# Charger le modèle et le scaler
model  = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Préparer un nouveau client (après prétraitement identique)
# X_new = ... (même pipeline que l'entraînement)

# Prédire
proba_churn = model.predict_proba(X_new)[:, 1]
print(f"Probabilité de churn : {proba_churn[0]:.2%}")
```

---

##  Technologies

![Python]
![Scikit-learn]
![XGBoost]
![Pandas]
![Jupyter]

| Catégorie | Outils |
|-----------|--------|
| Manipulation données | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Déséquilibre | `imbalanced-learn` (SMOTE) |
| Sauvegarde | `joblib` |

---

##  Insights Business

### Facteurs de risque (augmentent le churn)
1. **Contrat mensuel** → risque 3× plus élevé que contrat 2 ans
2. **Nouveaux clients** (tenure < 12 mois) → période critique
3. **Charges mensuelles élevées** → insatisfaction tarifaire
4. **Fibre optique** → clients plus volatils et exigeants
5. **Paiement par chèque électronique** → corrélé au churn

### Facteurs de rétention (réduisent le churn)
1. **Contrat 2 ans** → engagement = fidélité
2. **OnlineSecurity / TechSupport** → satisfaction accrue
3. **Ancienneté élevée** → effet loyauté

### Recommandations opérationnelles
-  Cibler les clients avec score de churn > 0.5 pour des actions proactives
-  Inciter les contrats mensuels à passer sur des offres annuelles
-  Promouvoir les services de sécurité et support dès l'onboarding
-  Monitorer intensément les clients durant leurs 12 premiers mois
-  Analyser les clients payant par chèque électronique

---

##  Méthodologie — Points clés

| Décision | Justification |
|----------|--------------|
| Stratification du split | Maintenir les proportions Churn/Non-Churn |
| SMOTE sur train uniquement | Éviter le data leakage |
| StandardScaler fit sur train | Éviter le data leakage |
| AUC-ROC comme métrique principale | Robuste au déséquilibre des classes |
| Pas de validation temporelle | Dataset sans dimension temporelle explicite |
| RandomizedSearchCV | Compromis exploration/coût de calcul |

---

##  Auteur

Projet réalisé seul pour perfectionnement 

---


