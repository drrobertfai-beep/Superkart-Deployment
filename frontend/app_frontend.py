import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(page_title="SuperKart Sales Forecaster", layout="wide")

st.title("🛒 SuperKart Store Sales Forecasting Portal")
st.write("Interactive interface for predicting product sales totals using the tuned Random Forest pipeline.")

# Configure backend URL
BACKEND_URL = "http://backend-api:7860"

tab1, tab2 = st.tabs(["📈 Real-time Prediction", "📁 Batch Prediction (CSV)"])

with tab1:
    st.header("Real-time Inference")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        product_weight = st.number_input("Product Weight (kg)", min_value=0.0, max_value=50.0, value=12.66)
        product_sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Allocated Display Area Ratio", min_value=0.0, max_value=1.0, value=0.027, format="%.4f")
        mrp = st.number_input("Product MRP ($)", min_value=0.0, max_value=500.0, value=117.08)
        
    with col2:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        
    with col3:
        product_id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
        store_age = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16)
        product_cat = st.selectbox("Product Category", ["Non Perishables", "Perishables", "Others"])

    if st.button("Predict Sales", key="single_predict"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_cat
        }
        
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Product Store Sales Total: **${result['prediction']:,.2f}**")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error occured')}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

with tab2:
    st.header("Batch File Inference")
    st.write("Upload a CSV file containing store data to get bulk predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.write("**File Preview:**")
        st.dataframe(df_preview.head())
        
        # Reset buffer pointer
        uploaded_file.seek(0)
        
        if st.button("Run Batch Predictions"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)
                
                if response.status_code == 200:
                    predictions = response.json()
                    # Map predictions back to the preview dataset
                    pred_list = [predictions[str(i)] for i in range(len(predictions))]
                    df_preview["Predicted_Sales"] = pred_list
                    
                    st.success("Batch Processing Complete!")
                    st.dataframe(df_preview)
                    
                    # Download link
                    csv_data = df_preview.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="SuperKart_Sales_Predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Batch Error: {response.json().get('error', 'Unknown backend error')}")
            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}")
