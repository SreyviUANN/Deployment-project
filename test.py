import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np

st.write("""
# The quality of water 
Using Random Forest        
""")

# Load the data
ad = pd.read_csv('water_potability.csv')

# Counter for generating unique keys
widget_counter = 0

# Function to get user input features
def user_input_features():
    global widget_counter
    widget_counter += 1

    ph = st.number_input('ph', value=ad['ph'].mean(), key=f'ph_input_{widget_counter}')
    Hardness = st.number_input('Hardness', value=ad['Hardness'].mean(), key=f'hardness_input_{widget_counter}')
    Solids = st.number_input('Solids', value=ad['Solids'].mean(), key=f'solids_input_{widget_counter}')
    Chloramines = st.number_input('Chloramines', value=ad['Chloramines'].mean(), key=f'chloramines_input_{widget_counter}')
    Sulfate = st.number_input('Sulfate', value=ad['Sulfate'].mean(), key=f'sulfate_input_{widget_counter}')
    Conductivity = st.number_input('Conductivity', value=ad['Conductivity'].mean(), key=f'conductivity_input_{widget_counter}')
    Organic_carbon = st.number_input('Organic_carbon', value=ad['Organic_carbon'].mean(), key=f'organic_carbon_input_{widget_counter}')
    Trihalomethanes = st.number_input('Trihalomethanes', value=ad['Trihalomethanes'].mean(), key=f'trihalomethanes_input_{widget_counter}')
    Turbidity = st.number_input('Turbidity', value=ad['Turbidity'].mean(), key=f'turbidity_input_{widget_counter}')
    
    data = {'ph': ph,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity}
    
    features = pd.DataFrame([data])
    return features

# Apply Z-score for outlier removal
def remove_outliers(df, features, threshold=2):
    z_scores = np.abs((df[features] - df[features].mean()) / df[features].std())
    df_no_outliers = df[(z_scores < threshold).all(axis=1)]
    return df_no_outliers

# Function to display input parameters and get predictions
def show_predictions():
    # Display input parameters
    st.subheader('Input Parameters:')
    input_data = user_input_features()
    st.write(input_data)

    # Combine user input with the original dataset
    df_combined = pd.concat([ad, input_data], ignore_index=True)

    # Fill missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df_imputed = df_combined.copy()
    df_imputed[['Sulfate', 'ph', 'Trihalomethanes']] = imputer.fit_transform(df_combined[['Sulfate', 'ph', 'Trihalomethanes']])

    # Check for missing values in the target variable
    if df_imputed['Potability'].isnull().any():
        #st.warning("Missing values found in the target variable 'Potability'. Handling missing values.")
        
        # Drop rows with missing values in the target variable
        df_imputed.dropna(subset=['Potability'], inplace=True)

    # Apply Z-score and remove outliers for all features
    all_features = ad.columns.to_list()
    all_features.remove('Potability')  # Assuming 'Potability' is your target variable
    df_no_outliers = remove_outliers(df_imputed, all_features)

    # Assuming 'Potability' is your target variable
    target = "Potability"
    features = df_no_outliers.columns.to_list()
    features.remove(target)

    # Train the model
    model = RandomForestClassifier()
    model.fit(df_no_outliers[features], df_no_outliers[target])

    # Make predictions for new data
    new_data_prediction = model.predict(input_data)
    
    # Display predictions
    st.subheader('Predicted Potability:')
    st.write(new_data_prediction)

# Show predictions
show_predictions()
