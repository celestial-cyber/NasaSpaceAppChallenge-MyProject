# app_interactive_pro.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
    df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\Project AstroScope\\KeplerExoplanetsDataset.csv")
    df['koi_kepmag'] = df['koi_kepmag'].fillna(df['koi_kepmag'].median())
    df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
    # Friendly column names
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

# --- Sidebar Filters ---
st.sidebar.header("Filter Candidates")

# Numeric filters
st.sidebar.subheader("Numeric Filters")
min_period, max_period = st.sidebar.slider("Orbital Period (days)", 
                                           float(df['Orbital Period (days)'].min()), 
                                           float(df['Orbital Period (days)'].max()), 
                                           (float(df['Orbital Period (days)'].min()), float(df['Orbital Period (days)'].max())))
min_mag, max_mag = st.sidebar.slider("Kepler Magnitude", 
                                     float(df['Kepler Mag'].min()), 
                                     float(df['Kepler Mag'].max()), 
                                     (float(df['Kepler Mag'].min()), float(df['Kepler Mag'].max())))
min_transit, max_transit = st.sidebar.slider("Transit Time (BKJD)", 
                                             float(df['Transit Time (BKJD)'].min()), 
                                             float(df['Transit Time (BKJD)'].max()), 
                                             (float(df['Transit Time (BKJD)'].min()), float(df['Transit Time (BKJD)'].max())))

# Boolean filters
st.sidebar.subheader("Flag Filters (Check for 1)")
not_transit_flag = st.sidebar.checkbox("Not Transit-Like")
stellar_eclipse_flag = st.sidebar.checkbox("Stellar Eclipse")
centroid_offset_flag = st.sidebar.checkbox("Centroid Offset")
ephemeris_contam_flag = st.sidebar.checkbox("Ephemeris Contamination")

# Apply filters
filtered_df = df[
    (df['Orbital Period (days)'] >= min_period) & (df['Orbital Period (days)'] <= max_period) &
    (df['Kepler Mag'] >= min_mag) & (df['Kepler Mag'] <= max_mag) &
    (df['Transit Time (BKJD)'] >= min_transit) & (df['Transit Time (BKJD)'] <= max_transit)
]

if not_transit_flag:
    filtered_df = filtered_df[filtered_df['Not Transit-Like'] == 1]
if stellar_eclipse_flag:
    filtered_df = filtered_df[filtered_df['Stellar Eclipse'] == 1]
if centroid_offset_flag:
    filtered_df = filtered_df[filtered_df['Centroid Offset'] == 1]
if ephemeris_contam_flag:
    filtered_df = filtered_df[filtered_df['Ephemeris Contamination'] == 1]

st.subheader(f"Filtered Dataset ({filtered_df.shape[0]} rows)")
st.dataframe(filtered_df.head(15))

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

# --- 3. Predictions & Top Candidates ---
pred_probs = rf_model.predict_proba(x_val)[:, 1]
results = x_val.copy()
results['Predicted Probability'] = pred_probs

top_candidates = results.sort_values('Predicted Probability', ascending=False).head(10)
st.subheader("Top 10 Predicted Exoplanets")

def color_probs(val):
    if val > 0.75:
        color = 'green'
    elif val > 0.5:
        color = 'orange'
    else:
        color = 'red'
    return f'background-color: {color}; color: white; font-weight: bold'

st.dataframe(top_candidates.style.applymap(color_probs, subset=['Predicted Probability']))

# Plot probabilities
fig1, ax1 = plt.subplots(figsize=(8,5))
colors = sns.color_palette("coolwarm", n_colors=10)
ax1.barh(top_candidates.index.astype(str), top_candidates['Predicted Probability'], color=colors)
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("Sample Index")
ax1.set_title("Top 10 Predicted Exoplanets")
ax1.invert_yaxis()
st.pyplot(fig1)

# --- 4. Global Feature Importance ---
perm_importance = permutation_importance(rf_model, x_val, y_val, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    'Feature': x_val.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

st.subheader("Global Feature Importance")
st.dataframe(feature_importance.style.bar(subset=['Importance'], color='lightblue'))

fig2, ax2 = plt.subplots(figsize=(8,5))
colors_feat = sns.color_palette("viridis", n_colors=len(feature_importance))
ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_feat)
ax2.set_xlabel("Permutation Importance")
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# --- 5. User Input for Prediction ---
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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
prob = rf_model.predict_proba(input_df)[:,1][0]

# Color-coded probability
if prob > 0.75:
    st.sidebar.success(f"High Probability! Predicted probability: {prob:.2f}")
elif prob > 0.5:
    st.sidebar.warning(f"Moderate Probability: {prob:.2f}")
else:
    st.sidebar.error(f"Low Probability: {prob:.2f}")
