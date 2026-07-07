import streamlit as st
import joblib
import numpy as np

# Step 1: Saved model aur scaler load karo
model = joblib.load("mall_kmeans_model.pkl")
scaler = joblib.load("mall_scaler.pkl")

# Step 2: Cluster numbers ko readable naam do (apne project ke mutabiq)
cluster_names = {
    0: "Older, Low Income, Low Spenders",
    1: "High Income, Low Spenders (Savers)",
    2: "Young, High Spenders",
    3: "Young, Moderate Spenders"
}

# Step 3: App ka title
st.title("Mall Customer Segmentation")
st.write("Customer ki details daalo, dekho woh kaunse group mein aata hai.")

# Step 4: User se input lena
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Step 5: Predict button
if st.button("Predict Cluster"):
    # Gender ko number mein badlo (training waqt jaisa encoding kiya tha)
    gender_num = 1 if gender == "Male" else 0

    # Input ko wahi format do jo model ko chahiye
    input_data = np.array([[gender_num, age, income, spending]])

    # Scale karo (training wale scaler se, wahi tareeqa jo humne seekha)
    input_scaled = scaler.transform(input_data)

    # Predict karo
    cluster = model.predict(input_scaled)[0]
    cluster_name = cluster_names[cluster]

    # Result dikhao
    st.success(f"Yeh customer Cluster {cluster} mein aata hai")
    st.info(f"Category: {cluster_name}")
