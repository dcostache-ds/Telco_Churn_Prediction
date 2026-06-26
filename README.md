# 📊 Customer Churn Prediction — End-to-End ML Project

Un proiect complet de Machine Learning care prezice dacă un client al unei companii de telecomunicații va pleca sau nu, bazat pe datele istorice ale acestuia.

---

## 🎯 Problema de Business

O companie de telecomunicații pierde clienți lunar. Costă de **5-10x mai mult** să atragi un client nou decât să păstrezi unul existent. Dacă putem prezice **din timp** cine va pleca, putem interveni cu oferte personalizate înainte să fie prea târziu.

**Soluția**: Un model ML antrenat pe 7043 de clienți reali care prezice probabilitatea de churn și simulează impactul diferitelor intervenții (reducere preț, schimbare contract).

---

## 📁 Structura Proiectului

```
churn-prediction/
├── data/
│   ├── Telco-Customer-Churn.csv      # Dataset original
│   └── processed/                     # Date procesate (generate automat)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Curatare, Encoding, Scalare, SMOTE
│   └── 03_modeling.ipynb              # Antrenare si comparare modele
├── dashboard/
│   └── app.py                         # Interfata Streamlit
├── models/                            # Generate automat dupa rularea notebooks
│   ├── best_model.pkl
│   ├── best_model_name.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── requirements.txt
```

---

## 🗂️ Dataset

**IBM Telco Customer Churn** — disponibil pe [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Proprietate | Valoare |
|---|---|
| Sursa | IBM / Kaggle |
| Clienti | 7.043 |
| Features | 21 coloane |
| Target | Churn (Yes/No) |
| Churn Rate | ~26.54% |

**Tipuri de features:**
- **Demografice**: gen, varsta, partener, dependenti
- **Servicii**: internet, telefonie, streaming, securitate
- **Contract**: tip contract, metoda de plata, facturare
- **Financiare**: taxa lunara, taxa totala, durata abonament

---

## 🔄 Pipeline ML

### 1. EDA (01_eda.ipynb)
- Analiza distributiei target-ului (26.54% churn)
- Vizualizarea relatiei dintre features si churn
- Identificarea problemelor din date

**Insight-uri principale:**
- Clientii noi (tenure mic) pleaca cel mai des — 50/50 in prima luna
- Contract lunar = risc maxim (42.71% churn vs 2.83% la 2 ani)
- Clientii cu factura mare (>80$/luna) au risc ridicat
- Fiber optic are cel mai mare churn rate din tipurile de internet

### 2. Preprocessing (02_preprocessing.ipynb)
| Pas | Ce face |
|---|---|
| Curatare | Spatii ascunse → NaN, valori lipsa, outlieri, duplicate, text inconsistent |
| Encoding | One-Hot Encoding cu `pd.get_dummies(drop_first=True)` |
| Scalare | `StandardScaler` pe coloanele numerice |
| Split | 80% train / 20% test cu `stratify=y` |
| SMOTE | Echilibrare clase: 27% → 50% (doar pe train) |

**Rezultat:** 30 features, 8.274 exemple de antrenare (dupa SMOTE)

### 3. Modeling (03_modeling.ipynb)
4 modele comparate pe aceleasi date:

| Model | ROC-AUC | F1-Score | Recall | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | **0.8517** | **0.6376** | **0.7413** | **0.7757** |
| LightGBM | 0.8334 | 0.6156 | 0.6853 | 0.7722 |
| XGBoost | 0.8326 | 0.6132 | 0.6827 | 0.7708 |
| Random Forest | 0.8308 | 0.5982 | 0.6213 | 0.7779 |

**Model ales: Logistic Regression** — cel mai bun ROC-AUC si Recall.

> **De ce Recall conteaza mai mult decat Accuracy?**
> Daca modelul zice mereu "clientul ramane", are 73% accuracy dar nu prinde niciun churner. Recall masoara cati churners reali am identificat corect — asta conteaza in business.

---

## 🖥️ Dashboard Streamlit

Interfata interactiva cu **simulare What-If** — cel mai util feature al proiectului.

### Cum functioneaza
1. Introduci datele unui client (contract, servicii, pret, durata)
2. Modelul calculeaza probabilitatea de churn
3. Interfata simuleaza automat scenarii alternative:
   - Ce se intampla daca schimbi contractul la 1 an / 2 ani?
   - Ce se intampla daca scazi pretul cu 10% / 20% / 30%?
   - Ce se intampla daca clientul sta inca 6 / 12 / 24 luni?
4. Afiseaza cea mai buna interventie posibila cu impact numeric

### Pornire
```bash
streamlit run dashboard/app.py
```

---

## ⚙️ Instalare si Rulare

### 1. Cloneaza proiectul
```bash
git clone https://github.com/username/churn-prediction.git
cd churn-prediction
```

### 2. Mediu virtual
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Instaleaza dependentele
```bash
pip install -r requirements.txt
```

### 4. Descarca datasetul
Descarca [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) de pe Kaggle si pune fisierul in `data/Telco-Customer-Churn.csv`.

### 5. Ruleaza notebooks (in ordine!)
```bash
jupyter notebook
# Ruleaza: 01_eda.ipynb → 02_preprocessing.ipynb → 03_modeling.ipynb
```

> ⚠️ Trebuie rulat in ordine — fiecare notebook genereaza fisierele necesare pentru urmatorul.

### 6. Porneste dashboard-ul
```bash
streamlit run dashboard/app.py
```

---

## 🛠️ Tech Stack

| Categorie | Librarii |
|---|---|
| **Data** | pandas, numpy |
| **Vizualizare** | matplotlib, seaborn |
| **ML** | scikit-learn, xgboost, lightgbm |
| **Imbalanced data** | imbalanced-learn (SMOTE) |
| **Salvare modele** | joblib |
| **Dashboard** | streamlit |

---

## 📈 Rezultate Cheie

- **74.13% Recall** — modelul identifica corect 3 din 4 clienti care vor pleca
- **85.17% ROC-AUC** — separare foarte buna intre churners si non-churners
- **Simulare What-If** — schimbarea contractului din lunar in 2 ani reduce riscul cu pana la 60%
- **Cel mai important factor**: tipul de contract (Month-to-month vs termen lung)

---

## 📝 Lectii invatate

- `pd.get_dummies()` pe un singur rand nu functioneaza corect la inferenta — solutia e constructia manuala a vectorului de features
- SMOTE se aplica **doar pe train**, niciodata pe test
- Recall > Accuracy pentru probleme cu clase dezechilibrate
- `stratify=y` in train_test_split e critic pentru a pastra proportia claselor

---

## 👤 Autor

Proiect realizat ca parte a procesului de invatare Data Science / Machine Learning.
