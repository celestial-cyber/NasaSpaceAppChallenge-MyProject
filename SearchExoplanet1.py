# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

st.set_page_config(page_title="AstroScope", page_icon="🔭")
st.title("🔭 AstroScope: Intelligent Exoplanet Discovery Dashboard")

st.markdown("""
This dashboard predicts the probability of confirmed exoplanets using a Random Forest model trained on Kepler dataset.
""")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\Project AstroScope\\KeplerExoplanetsDataset.csv")
    # Fill missing values
    df['koi_kepmag'] = df['koi_kepmag'].fillna(df['koi_kepmag'].median())
    # Create label
    df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
    return df

df = load_data()
st.subheader("Raw Dataset (first 5 rows)")
st.dataframe(df.head())

# --- 2. Preprocessing ---
num_cols = ['koi_period', 'koi_time0bk', 'koi_kepmag']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df[['koi_period', 'koi_time0bk', 'koi_kepmag', 'koi_fpflag_nt', 
        'koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec']]
y = df['label']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# --- 3. Train Random Forest ---
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

rf_model = train_model(X_train, y_train)

# --- 4. Predictions & Probabilities ---
pred_probs = rf_model.predict_proba(x_val)[:, 1]
results = x_val.copy()
results['predicted_prob'] = pred_probs

st.subheader("Top 10 Predicted Exoplanets")
top_candidates = results.sort_values('predicted_prob', ascending=False).head(10)
st.dataframe(top_candidates)

# Plot top predicted probabilities
st.subheader("Top 10 Prediction Probabilities")
fig1, ax1 = plt.subplots()
ax1.barh(top_candidates.index.astype(str), top_candidates['predicted_prob'])
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("Sample Index")
ax1.set_title("Top 10 Predicted Exoplanets")
ax1.invert_yaxis()
st.pyplot(fig1)

# --- 5. Feature Importance ---
perm_importance = permutation_importance(rf_model, x_val, y_val, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    'feature': x_val.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

st.subheader("Feature Importance (Global)")
st.dataframe(feature_importance)

# Plot feature importance
fig2, ax2 = plt.subplots()
ax2.barh(feature_importance['feature'], feature_importance['importance'])
ax2.set_xlabel("Permutation Importance")
ax2.set_title("Global Feature Importance")
st.pyplot(fig2)

# --- 6. Predict for custom input ---
st.subheader("Predict Probability for Custom Input")

def user_input_features():
    koi_period = st.number_input("koi_period", value=0.01, format="%.6f")
    koi_time0bk = st.number_input("koi_time0bk", value=0.02, format="%.6f")
    koi_kepmag = st.number_input("koi_kepmag", value=0.5, format="%.6f")
    koi_fpflag_nt = st.selectbox("koi_fpflag_nt", [0,1])
    koi_fpflag_ss = st.selectbox("koi_fpflag_ss", [0,1])
    koi_fpflag_co = st.selectbox("koi_fpflag_co", [0,1])
    koi_fpflag_ec = st.selectbox("koi_fpflag_ec", [0,1])
    
    data = {
        'koi_period': koi_period,
        'koi_time0bk': koi_time0bk,
        'koi_kepmag': koi_kepmag,
        'koi_fpflag_nt': koi_fpflag_nt,
        'koi_fpflag_ss': koi_fpflag_ss,
        'koi_fpflag_co': koi_fpflag_co,
        'koi_fpflag_ec': koi_fpflag_ec
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
prob = rf_model.predict_proba(input_df)[:,1][0]
st.write(f"Predicted probability of being a confirmed exoplanet: **{prob:.2f}**")
