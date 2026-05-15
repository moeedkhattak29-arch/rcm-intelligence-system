"""
DenialShield — Model Training Script
Run this ONCE to retrain all models on your local machine.
This fixes the sklearn version mismatch error.

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_absolute_error, mean_squared_error,
                              r2_score, classification_report)
import joblib
import warnings
import sklearn
warnings.filterwarnings('ignore')

print("=" * 55)
print("  DenialShield — Model Training Script")
print(f"  sklearn version: {sklearn.__version__}")
print("=" * 55)

# =============================================
# MODEL 1: DENIAL PREDICTION
# =============================================
print("\n[1/4] Training Denial Prediction model...")

df1 = pd.read_csv("rcm_claims_data__1_.csv")
df1['Reason Code'] = df1['Reason Code'].fillna('Unknown')

X1 = df1[['CPT Code', 'Diagnosis Code', 'Insurance Type',
           'AR Status', 'Reason Code', 'Days in AR']]
y1 = df1['denial']

cat1 = ['Diagnosis Code', 'Insurance Type', 'AR Status', 'Reason Code']
num1 = ['CPT Code', 'Days in AR']

pre1 = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler())
    ]), num1),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat1)
])

model1 = Pipeline([
    ('pre', pre1),
    ('clf', RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ))
])

X1tr, X1te, y1tr, y1te = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1
)
model1.fit(X1tr, y1tr)
y1p  = model1.predict(X1te)
y1pr = model1.predict_proba(X1te)[:, 1]

print(f"   Accuracy : {accuracy_score(y1te, y1p):.4f}")
print(f"   F1 Score : {f1_score(y1te, y1p):.4f}")
print(f"   AUC-ROC  : {roc_auc_score(y1te, y1pr):.4f}")
joblib.dump(model1, "denial_prediction_model.pkl")
print("   ✅ Saved: denial_prediction_model.pkl")

# =============================================
# MODEL 2: DENIAL REASON
# =============================================
print("\n[2/4] Training Denial Reason model...")

df2 = pd.read_csv("rcm_full_dataset__1_.csv")
df2_denied = df2[df2['is_denied'] == 1].copy()
df2_denied['denial_reason'] = df2_denied['denial_reason'].fillna('CO-16')
df2_denied['modifier']      = df2_denied['modifier'].fillna('None')

le = LabelEncoder()
df2_denied['denial_reason_encoded'] = le.fit_transform(df2_denied['denial_reason'])
print(f"   Classes  : {list(le.classes_)}")

X2 = df2_denied[['payer', 'cpt', 'icd10', 'modifier',
                  'amount', 'age', 'prior_auth', 'doc_score']]
y2 = df2_denied['denial_reason_encoded']

cat2 = ['payer', 'icd10', 'modifier']
num2 = ['cpt', 'amount', 'age', 'prior_auth', 'doc_score']

pre2 = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler())
    ]), num2),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat2)
])

model2 = Pipeline([
    ('pre', pre2),
    ('clf', RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ))
])

X2tr, X2te, y2tr, y2te = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2.fit(X2tr, y2tr)
y2p = model2.predict(X2te)

print(f"   Accuracy : {accuracy_score(y2te, y2p):.4f}")
print(f"   F1 Score : {f1_score(y2te, y2p, average='weighted'):.4f}")
joblib.dump(model2, "denial_reason_model.pkl")
joblib.dump(le,     "denial_reason_label_encoder.pkl")
print("   ✅ Saved: denial_reason_model.pkl")
print("   ✅ Saved: denial_reason_label_encoder.pkl")

# =============================================
# MODEL 3: APPEAL SUCCESS
# =============================================
print("\n[3/4] Training Appeal Success model...")

df3 = pd.read_csv("Appeal_dataset.csv")
X3 = df3[['patient_age', 'gender', 'claim_type', 'insurance_type',
           'denial_reason', 'ar_days', 'prior_appeals']]
y3 = df3['success']

cat3 = ['gender', 'claim_type', 'insurance_type', 'denial_reason']
num3 = ['patient_age', 'ar_days', 'prior_appeals']

pre3 = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler())
    ]), num3),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat3)
])

model3 = Pipeline([
    ('pre', pre3),
    ('clf', RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ))
])

X3tr, X3te, y3tr, y3te = train_test_split(
    X3, y3, test_size=0.2, random_state=42, stratify=y3
)
model3.fit(X3tr, y3tr)
y3p  = model3.predict(X3te)
y3pr = model3.predict_proba(X3te)[:, 1]

print(f"   Accuracy : {accuracy_score(y3te, y3p):.4f}")
print(f"   F1 Score : {f1_score(y3te, y3p):.4f}")
print(f"   AUC-ROC  : {roc_auc_score(y3te, y3pr):.4f}")
joblib.dump(model3, "appeal_success_model.pkl")
print("   ✅ Saved: appeal_success_model.pkl")

# =============================================
# MODEL 4: AR DAYS FORECAST
# =============================================
print("\n[4/4] Training AR Days Forecast model...")

df4 = pd.read_csv("healthcare_claims.csv")
X4 = df4[['Payer', 'CPT_Code', 'Claim_Amount', 'Denial_History',
           'Resubmission_Count', 'Patient_Responsibility', 'Billing_Lag']]
y4 = df4['AR_Days']

cat4 = ['Payer']
num4 = ['CPT_Code', 'Claim_Amount', 'Denial_History',
        'Resubmission_Count', 'Patient_Responsibility', 'Billing_Lag']

pre4 = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler())
    ]), num4),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat4)
])

model4 = Pipeline([
    ('pre', pre4),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
])

X4tr, X4te, y4tr, y4te = train_test_split(X4, y4, test_size=0.2, random_state=42)
model4.fit(X4tr, y4tr)
y4p = model4.predict(X4te)

print(f"   MAE      : {mean_absolute_error(y4te, y4p):.2f} days")
print(f"   RMSE     : {np.sqrt(mean_squared_error(y4te, y4p)):.2f} days")
print(f"   R²       : {r2_score(y4te, y4p):.4f}")
joblib.dump(model4, "ar_days_model.pkl")
print("   ✅ Saved: ar_days_model.pkl")

# =============================================
# DONE
# =============================================
print("\n" + "=" * 55)
print("  ✅ All 4 models trained and saved successfully!")
print("  Now run:  python -m streamlit run app.py")
print("=" * 55)
