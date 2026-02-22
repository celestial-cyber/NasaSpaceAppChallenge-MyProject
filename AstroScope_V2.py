# app_interactive_pro.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AstroScope", page_icon="🔭", layout="wide")
st.title("🔭 AstroScope: Intelligent Exoplanet Discovery Dashboard")

st.markdown("""
Explore exoplanet candidates and predict probabilities of confirmed exoplanets using a Random Forest model.
Use the filters and input section to interactively explore the dataset.
""")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\NasaSpaceAppChallenge-MyProject\\KeplerExoplanetsDataset.csv")
    df['koi_kepmag'] = df['koi_kepmag'].fillna(df['koi_kepmag'].median())
    df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
    df.rename(columns={
        'koi_period': 'Orbital Period (days)',
        'koi_time0bk': 'Transit Time (BKJD)',
        'koi_kepmag': 'Kepler Mag',
        'koi_fpflag_nt': 'Not Transit-Like',
        'koi_fpflag_ss': 'Stellar Eclipse',
        'koi_fpflag_co': 'Centroid Offset',
        'koi_fpflag_ec': 'Ephemeris Contamination'
    }, inplace=True)
    return df

df = load_data()

# --- 2. Preprocessing & Train Model ---
num_cols = ['Orbital Period (days)', 'Transit Time (BKJD)', 'Kepler Mag']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df[['Orbital Period (days)', 'Transit Time (BKJD)', 'Kepler Mag', 
        'Not Transit-Like', 'Stellar Eclipse', 'Centroid Offset', 'Ephemeris Contamination']]
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

rf_model = train_model(X_train, y_train)

# --- 3. Confusion Matrix ---
st.subheader("Confusion Matrix")

y_pred = rf_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Confirmed', 'Confirmed'],
            yticklabels=['Not Confirmed', 'Confirmed'])
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# --- 4. Classification Report ---
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# --- 5. Global Feature Importance ---
perm_importance = permutation_importance(rf_model, x_val, y_val, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    'Feature': x_val.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

st.subheader("Global Feature Importance")
st.dataframe(feature_importance)

fig2, ax2 = plt.subplots()
ax2.barh(feature_importance['Feature'], feature_importance['Importance'])
ax2.set_xlabel("Permutation Importance")
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# --- 6. User Input Prediction ---
st.sidebar.header("Predict Your Exoplanet Candidate")

def user_input_features():
    orbital_period = st.sidebar.number_input("Orbital Period (days)", value=0.01, format="%.6f")
    transit_time = st.sidebar.number_input("Transit Time (BKJD)", value=0.02, format="%.6f")
    kepler_mag = st.sidebar.number_input("Kepler Mag", value=0.5, format="%.6f")
    not_transit_like = st.sidebar.selectbox("Not Transit-Like Flag", [0,1])
    stellar_eclipse = st.sidebar.selectbox("Stellar Eclipse Flag", [0,1])
    centroid_offset = st.sidebar.selectbox("Centroid Offset Flag", [0,1])
    ephemeris_contamination = st.sidebar.selectbox("Ephemeris Contamination Flag", [0,1])
    
    data = {
        'Orbital Period (days)': orbital_period,
        'Transit Time (BKJD)': transit_time,
        'Kepler Mag': kepler_mag,
        'Not Transit-Like': not_transit_like,
        'Stellar Eclipse': stellar_eclipse,
        'Centroid Offset': centroid_offset,
        'Ephemeris Contamination': ephemeris_contamination
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
prob = rf_model.predict_proba(input_df)[:,1][0]

if prob > 0.75:
    st.sidebar.success(f"High Probability! Predicted probability: {prob:.2f}")
elif prob > 0.5:
    st.sidebar.warning(f"Moderate Probability: {prob:.2f}")
else:
    st.sidebar.error(f"Low Probability: {prob:.2f}")
