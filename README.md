# 📊 Customer Churn Prediction — End-to-End ML Project

A complete Machine Learning project that predicts whether a telecom customer will churn (leave the company) based on their historical data.

---

## 🎯 Business Problem

A telecommunications company loses customers every month. It costs **5-10x more** to acquire a new customer than to retain an existing one. If we can predict **in advance** who is likely to leave, we can intervene with personalized offers before it's too late.

**Solution**: An ML model trained on 7,043 real customers that predicts churn probability and simulates the impact of different interventions (price reduction, contract upgrade).

---

## 📁 Project Structure

```
churn-prediction/
├── data/
│   ├── Telco-Customer-Churn.csv      # Original dataset
│   └── processed/                     # Processed data (auto-generated)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Cleaning, Encoding, Scaling, SMOTE
│   └── 03_modeling.ipynb              # Model training and comparison
├── dashboard/
│   └── app.py                         # Streamlit interface
├── models/                            # Auto-generated after running notebooks
│   ├── best_model.pkl
│   ├── best_model_name.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── requirements.txt
```

---

## 🗂️ Dataset

**IBM Telco Customer Churn** — available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Source | IBM / Kaggle |
| Customers | 7,043 |
| Features | 21 columns |
| Target | Churn (Yes/No) |
| Churn Rate | ~26.54% |

**Feature types:**
- **Demographics**: gender, age, partner, dependents
- **Services**: internet, phone, streaming, online security
- **Contract**: contract type, payment method, billing
- **Financial**: monthly charges, total charges, tenure

---

## 🔄 ML Pipeline

### 1. EDA (01_eda.ipynb)
- Target distribution analysis (26.54% churn)
- Feature vs churn visualizations
- Data quality issues identification

**Key insights:**
- New customers (low tenure) churn the most — 50/50 chance in the first month
- Month-to-month contract = highest risk (42.71% churn vs 2.83% for 2-year contracts)
- Customers with high monthly charges (>$80) have elevated risk
- Fiber optic has the highest churn rate among internet service types

### 2. Preprocessing (02_preprocessing.ipynb)
| Step | What it does |
|---|---|
| Cleaning | Hidden spaces → NaN, missing values, outliers, duplicates, inconsistent text |
| Encoding | One-Hot Encoding with `pd.get_dummies(drop_first=True)` |
| Scaling | `StandardScaler` on numerical columns |
| Split | 80% train / 20% test with `stratify=y` |
| SMOTE | Class balancing: 27% → 50% (train set only) |

**Result:** 30 features, 8,274 training samples (after SMOTE)

### 3. Modeling (03_modeling.ipynb)
4 models compared on the same data:

| Model | ROC-AUC | F1-Score | Recall | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | **0.8517** | **0.6376** | **0.7413** | **0.7757** |
| LightGBM | 0.8334 | 0.6156 | 0.6853 | 0.7722 |
| XGBoost | 0.8326 | 0.6132 | 0.6827 | 0.7708 |
| Random Forest | 0.8308 | 0.5982 | 0.6213 | 0.7779 |

**Chosen model: Logistic Regression** — best ROC-AUC and Recall.

> **Why Recall matters more than Accuracy?**
> A model that always predicts "customer stays" achieves 73% accuracy but catches zero churners. Recall measures how many actual churners we correctly identified — that's what matters in business.

---

## 🖥️ Streamlit Dashboard

Interactive interface with **What-If simulation** — the most useful feature of this project.

### How it works
1. Enter a customer's data (contract, services, price, tenure)
2. The model calculates the churn probability
3. The interface automatically simulates alternative scenarios:
   - What happens if you upgrade the contract to 1 year / 2 years?
   - What happens if you reduce the price by 10% / 20% / 30%?
   - What happens if the customer stays for another 6 / 12 / 24 months?
4. Displays the best possible intervention with numeric impact

### Run the dashboard
```bash
streamlit run dashboard/app.py
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/username/churn-prediction.git
cd churn-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place the file in `data/Telco-Customer-Churn.csv`.

### 5. Run the notebooks (in order!)
```bash
jupyter notebook
# Run in order: 01_eda.ipynb → 02_preprocessing.ipynb → 03_modeling.ipynb
```

> ⚠️ Must be run in order — each notebook generates files needed by the next one.

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| **Data** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **ML** | scikit-learn, xgboost, lightgbm |
| **Imbalanced data** | imbalanced-learn (SMOTE) |
| **Model persistence** | joblib |
| **Dashboard** | streamlit |

---

## 📈 Key Results

- **74.13% Recall** — the model correctly identifies 3 out of 4 customers who will churn
- **85.17% ROC-AUC** — strong separation between churners and non-churners
- **What-If simulation** — switching from a monthly to a 2-year contract reduces churn risk by up to 60%
- **Most important factor**: contract type (Month-to-month vs long-term)

---

## 📝 Key Learnings

- `pd.get_dummies()` on a single row doesn't work correctly at inference time — the solution is manually building the feature vector
- SMOTE must be applied **only on the train set**, never on the test set
- Recall > Accuracy for imbalanced class problems
- `stratify=y` in `train_test_split` is critical to maintain class proportions in both sets

---

## 👤 Author

Project built as part of a Data Science / Machine Learning learning journey.
