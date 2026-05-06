
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from openai import OpenAI
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


MODEL_PATH = Path(__file__).parent / "superkart_sales_model_v1_0.joblib"
model = joblib.load(MODEL_PATH)

st.title("SuperKart Sales Prediction App")
st.subheader("Enter Product & Store Details")


# Input fields
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=10.0
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=100.0
)

Store_Size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
)

Product_Id_char = st.selectbox(
    "Product ID Character",
    ["FD", "DR", "NC"]
)

Store_Age_Years = st.number_input(
    "Store Age Years",
    min_value=0,
    value=5
)

Product_Type_Category = st.selectbox(
    "Product Type Category",
    ["Food", "Non-Consumable", "Drinks"]
)


if st.button("Predict Sales"):

    sample = pd.DataFrame([{
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char,
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category
    }])

    prediction = model.predict(sample)

    st.session_state["prediction"] = prediction
    st.session_state["sample"] = sample
    
if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    sample = st.session_state["sample"]

    st.markdown(f"""
    <div style="text-align: center; margin-top: 30px;">
        <h3>💰 Predicted Sales</h3>
        <h1>${prediction[0]:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.caption("⬇ Ask the assistant how to improve this number")
    st.markdown("---")

    with st.expander("🤖 Sales Strategy Assistant", expanded=True):

        # 🧠 Initialize memory ONCE
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # User input
        question = st.text_input(
            "Ask about this prediction",
            value="Why is this sales prediction this value and how can I improve it?"
        )

        # Button
        if st.button("Ask Assistant"):

            # Add user message to memory
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            # Build prompt WITH your data
            prompt = f"""
You are a retail sales expert.

A machine learning model predicted sales of ${prediction[0]:,.2f}.

Input data:
{sample.to_dict(orient="records")[0]}

Explain why this prediction might be this value.
Then give 3 specific ways to increase sales.
Keep it clear, short, and actionable.

User question: {question}
"""

            # Add system context
            messages = [
                {"role": "system", "content": "You are a helpful retail sales strategist."}
        ] + st.session_state.chat_history + [
            {"role": "user", "content": prompt}
        ]

            # Call model
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=messages
            )

            answer = response.output_text

            # Save assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            # Show answer
            st.write(answer)