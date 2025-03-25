import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Read the CSV file
@st.cache_data
def load_data():
    df = pd.read_csv('car_data.csv')
    return df

# Title and description of the app
st.set_page_config(
    page_title="CarPricePredictor", 
    # page_icon="🌍", 
    layout="wide", 
    initial_sidebar_state="expanded")

st.title("Car Price Predictor")
st.markdown("""
Enter your car's details to get an estimated price.
""")

# Step 1: Upload the trained model file
@st.cache_resource  # Cache the model to avoid reloading on every rerun
def load_model():
    with open('tree_model.pkl', 'rb') as f:
        tree_model = joblib.load(f)
    return tree_model

if 'tree_model' not in st.session_state:
    st.session_state.tree_model = load_model()

# Uploading target encoder
with open('target_encoder.pkl', 'rb') as f:
    encoder = joblib.load(f)

# Uploading Scaler
with open('scaler.pkl','rb') as f:
    scaler = joblib.load(f)

# Define input fields for the user
st.subheader("Car Specifications")

year = st.slider('Year of Manufacture', min_value=1996, max_value=2020, step=1)
km_driven = st.number_input('Kilometers Driven', min_value=100, max_value=3800000, value=17512)
mileage = st.number_input('Mileage (km/l)', min_value=4.0, max_value=120.0, value= 23.59, format="%.2f", step=0.01)
engine = st.number_input('Engine Capacity (cc)', min_value=0.0, max_value=6752.0, value= 1364.0, format="%.2f")
max_power = st.number_input('Max Power (bhp)', min_value= 5.0, max_value= 626.0, value= 103.52)
age = st.slider('Car Age (Years)', min_value= 2, max_value= 31, value= 4, step=1)

st.markdown("---")

st.subheader("Car Type & Brand")

# Main Streamlit app
def main():
    # Load the data
    df = load_data()
    
    # Get unique makes (brands)
    unique_makes = sorted(df['make'].unique())
    
    # First dropdown for selecting make
    selected_make = st.selectbox('Select Car Make', [''] + list(unique_makes))
    
    # Initialize selected_model as None
    selected_model = None
    
    # If a make is selected, filter models for that make
    if selected_make:
        # Filter models based on the selected make
        filtered_models = sorted(df[df['make'] == selected_make]['model'].unique())
        
        # Second dropdown for selecting model
        selected_model = st.selectbox('Select Car Model', filtered_models)
        
    else:
        st.write("Please select a car brand first.")
    
    # Return the selected values
    return selected_make, selected_model

# Run the app and get the selected make and model
selected_make, selected_model = main()

st.markdown("---")

st.subheader("Seller Information")
seller_type = st.radio("Seller Type", ["Individual", "Trustmark Dealer", "None"])

st.markdown("---")

st.subheader("Fuel Type")
fuel_type = st.radio("Fuel Type", ["Diesel", "Electric", "LPG", "Petrol"])

st.markdown("---")

st.subheader("Transmission Type")
transmission = st.radio("Transmission", ["Manual", "Automatic"])

st.markdown("---")

st.subheader("Seating Capacity")
seating_capacity = st.radio("Seating Capacity", ["5", "More than 5"])

st.markdown("---")

# Button to trigger prediction
if st.button("Predict"):

    # Converting seller type to binary columns
    if seller_type == "Individual":
        individual = 1
        trustmark_dealer = 0
    elif seller_type == "Trustmark Dealer":
        individual = 0
        trustmark_dealer = 1
    else:  # seller_type == "None"
        individual = 0
        trustmark_dealer = 0

    # Converting fuel type to binary columns
    diesel = 1 if fuel_type == "Diesel" else 0
    electric = 1 if fuel_type == "Electric" else 0
    lpg = 1 if fuel_type == "LPG" else 0
    petrol = 1 if fuel_type == "Petrol" else 0

    # Converting transmission type to binary column
    manual = 1 if transmission == "Manual" else 0
    automatic = 1 if transmission == "Automatic" else 0

    # Converting seating capacity to binary columns
    five_seater = 1 if seating_capacity == "5" else 0
    more_than_five_seater = 1 if seating_capacity == "More than 5" else 0

    # Creating a DataFrame from the input data
    input_data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "age": [age],
        "make": [selected_make],
        "model": [selected_model],
        "Individual": [individual],
        "Trustmark Dealer": [trustmark_dealer],
        "Diesel": [diesel],
        "Electric": [electric],
        "LPG": [lpg],
        "Petrol": [petrol],
        "Manual": [manual],
        "5": [five_seater],
        ">5": [more_than_five_seater]
    })
    
    # Display the input data for debugging purposes
    st.write("Input Data:")
    st.write(input_data)
    
    # Apply target encoding and scaling
    
    # Apply target encoding to both 'make' and 'model' columns
    input_data[['make', 'model']] = encoder.transform(input_data[['make', 'model']]) # Apply target encoding

    input_data_scaled = scaler.transform(input_data)  # Apply scaling
    
    # Make predictions using the loaded model
    predicted_price = st.session_state.tree_model.predict(input_data_scaled)[0]

    
    # Display prediction
    st.success("Prediction Successful!")
    st.markdown(f"<h1>Predicted Price: ₹{predicted_price:.2f} Lakhs</h1>", unsafe_allow_html=True)

    