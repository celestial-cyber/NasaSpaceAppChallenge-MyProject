# app_interactive.py

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
Explore exoplanet candidates and predict probabilities of confirmed exoplanets using a Random Forest model.
""")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\Project AstroScope\\KeplerExoplanetsDataset.csv")
    df['koi_kepmag'] = df['koi_kepmag'].fillna(df['koi_kepmag'].median())
    df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Candidates")
min_period = st.sidebar.number_input("Min koi_period", value=float(df['koi_period'].min()))
max_period = st.sidebar.number_input("Max koi_period", value=float(df['koi_period'].max()))
min_mag = st.sidebar.number_input("Min koi_kepmag", value=float(df['koi_kepmag'].min()))
max_mag = st.sidebar.number_input("Max koi_kepmag", value=float(df['koi_kepmag'].max()))

filtered_df = df[(df['koi_period'] >= min_period) & (df['koi_period'] <= max_period) &
                 (df['koi_kepmag'] >= min_mag) & (df['koi_kepmag'] <= max_mag)]

st.subheader(f"Filtered Dataset ({filtered_df.shape[0]} rows)")
st.dataframe(filtered_df.head(10))

# --- 2. Preprocessing ---
num_cols = ['koi_period', 'koi_time0bk', 'koi_kepmag']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df[['koi_period', 'koi_time0bk', 'koi_kepmag', 'koi_fpflag_nt', 
        'koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec']]
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# --- 3. Train Model ---
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

rf_model = train_model(X_train, y_train)

# --- 4. Predictions & Top Candidates ---
pred_probs = rf_model.predict_proba(x_val)[:, 1]
results = x_val.copy()
results['predicted_prob'] = pred_probs

top_candidates = results.sort_values('predicted_prob', ascending=False).head(10)
st.subheader("Top 10 Predicted Exoplanets")
st.dataframe(top_candidates)

# Plot probabilities
fig1, ax1 = plt.subplots()
ax1.barh(top_candidates.index.astype(str), top_candidates['predicted_prob'])
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("Sample Index")
ax1.set_title("Top 10 Predicted Exoplanets")
ax1.invert_yaxis()
st.pyplot(fig1)

# --- 5. Global Feature Importance ---
perm_importance = permutation_importance(rf_model, x_val, y_val, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    'feature': x_val.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

st.subheader("Global Feature Importance")
st.dataframe(feature_importance)

fig2, ax2 = plt.subplots()
ax2.barh(feature_importance['feature'], feature_importance['importance'])
ax2.set_xlabel("Permutation Importance")
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# --- 6. User Input for Prediction ---
st.sidebar.header("Predict Your Exoplanet Candidate")

def user_input_features():
    koi_period = st.sidebar.number_input("koi_period", value=0.01, format="%.6f")
    koi_time0bk = st.sidebar.number_input("koi_time0bk", value=0.02, format="%.6f")
    koi_kepmag = st.sidebar.number_input("koi_kepmag", value=0.5, format="%.6f")
    koi_fpflag_nt = st.sidebar.selectbox("koi_fpflag_nt", [0,1])
    koi_fpflag_ss = st.sidebar.selectbox("koi_fpflag_ss", [0,1])
    koi_fpflag_co = st.sidebar.selectbox("koi_fpflag_co", [0,1])
    koi_fpflag_ec = st.sidebar.selectbox("koi_fpflag_ec", [0,1])
    
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
st.sidebar.write(f"Predicted probability of being a confirmed exoplanet: **{prob:.2f}**")
