import streamlit as st
import joblib
import pandas as pd

# Loading the saved model, encoder, scaler, and columns.
model = joblib.load("SVM_Model.pkl")
scaler = joblib.load("PowerTransformer_Scaler.pkl") 
model_columns = joblib.load("Model_Columns.pkl")    
le = joblib.load("Label_Encoder.pkl")

# UI set up. 
st.set_page_config(page_title = "Dry Bean Type Classification", page_icon = "🌱", layout = "centered")
st.title("🌱 Dry Bean Type Classification")

st.write("Please Input the following Features to Classify the type of Dry Bean:")

# User input fields. 
Area = st.number_input("Area", min_value = 0.0, step = 0.01)
Perimeter = st.number_input("Perimeter", min_value = 0.0, step = 0.01)
MajorAxisLength = st.number_input("Major Axis Length", min_value = 0.0, step = 0.01)
MinorAxisLength = st.number_input("Minor Axis Length", min_value = 0.0, step = 0.01)
AspectRatio = st.number_input("Aspect Ratio", min_value = 0.0, step = 0.01)
Eccentricity = st.number_input("Eccentricity", min_value = 0.0, step = 0.01)
ConvexArea = st.number_input("Convex Area", min_value = 0.0, step = 0.01)
EquivDiameter = st.number_input("Equivalent Diameter", min_value = 0.0, step = 0.01)
Extent = st.number_input("Extent", min_value = 0.0, step = 0.01)
Solidity = st.number_input("Solidity", min_value = 0.0, step = 0.01)
Roundness = st.number_input("Roundness", min_value = 0.0, step = 0.01)
Compactness = st.number_input("Compactness", min_value = 0.0, step = 0.01)
ShapeFactor1 = st.number_input("Shape Factor 1", min_value = 0.0, step = 0.01)
ShapeFactor2 = st.number_input("Shape Factor 2", min_value = 0.0, step = 0.01)
ShapeFactor3 = st.number_input("Shape Factor 3", min_value = 0.0, step = 0.01)
ShapeFactor4 = st.number_input("Shape Factor 4", min_value = 0.0, step = 0.01)

# Preparing the input data for prediction.  
if st.button("Predict Bean Type"):
    
    # Creating the input values as a list of lists (2D list) in the same order as saved in the model_columns.
    input_values = [[Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4]]
    
    # Creating a DataFrame for the input data.
    input_data = pd.DataFrame(input_values, columns = model_columns)
    
    # Scaling the input data.
    input_data_scaled = scaler.transform(input_data)

    # Predicting the class label for the input data.
    prediction_encoded = model.predict(input_data_scaled)

    # Decoding the predicted class label back to the original class name.
    prediction_decoded = le.inverse_transform(prediction_encoded)  

    # Displaying the predicted class label to the user.
    st.success(f"The Predicted type of Dry Bean is: {prediction_decoded[0]}")