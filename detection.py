import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved BERT model and tokenizer (this will point to the model directory after downloading)
model_path = r'bert_model'  # Ensure the correct path
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Preprocess the text for BERT
def transform_text(text):
    # Tokenize and encode the input
    encodings = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,  # Adjust this based on your training parameters
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']

# Set up the page configuration
st.set_page_config(page_title="Email Fraud Detection")

# Adding custom CSS for styling
st.markdown(
    """
    <style>
    * {
        box-sizing: border-box;
    }
    .stApp {
        background-image: url("https://i.pinimg.com/736x/6e/46/da/6e46da2c1712b7daaba49f78988221a4.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    .custom-title {
        float: left;
        margin-left: -24rem;
        font-weight: 1000;
        font-size: 3rem;
        font-family: lucida;
    }
    .left-section {
        float: left;
        margin-left: -24rem;
        font-weight: 500;
        font-size: 1.7rem;
        font-family: lucida;
    }
    .stTextArea {
        margin-left: -24rem !important;
        width: 600px !important;
    }
    .stButton>button {
        margin-left: -24rem !important;
        background-color: #008CBA;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: color 0.3s ease, background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005f73;
    }
    .result-container {
        float: left;
        font-size: large;
    }
    .result-container1 {
        float: left;
        margin-left: -24rem;
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
        padding: 20px;
        margin-top: 1rem;
        width: 100%;
        max-width: 600px;
        font-size: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .percentage {
        float: left;
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
        padding: 20px;
        margin-top: -1rem;
        width: 100%;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown('<div class="custom-title">📧 Email Fraud Detection</div>', unsafe_allow_html=True)

# Create layout with two columns
col1, col2 = st.columns([2, 2])

with col1:
    # Input section title
    st.markdown('<div class="left-section">Detect whether an email is Fraudulent or Non-fraudulent.</div>', unsafe_allow_html=True)

    # Text input for the email content
    input_sms = st.text_area("Enter the email text below:")

    # Analyze email button and result display
    if st.button('Analyze Email'):
        if input_sms.strip() == "":
            # Alert for empty input
            st.markdown(
                '<div class="custom-alert">⚠️ Please enter text before analyzing.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analyzing..."):
                # Preprocess the input
                input_ids, attention_mask = transform_text(input_sms)

                # Move tensors to the appropriate device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                model.to(device)

                # Get model prediction
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()

                # Calculate confidence for each class
                confidence_non_fraudulent = probabilities[0][0]
                confidence_fraudulent = probabilities[0][1]

                # Store confidence scores for later use
                st.session_state['confidence_non_fraudulent'] = confidence_non_fraudulent
                st.session_state['confidence_fraudulent'] = confidence_fraudulent

                # Determine result based on threshold
                threshold = 0.5
                result = "Fraudulent" if confidence_fraudulent >= threshold else "Non-Fraudulent"

                # Display result in a styled box
                st.markdown(
                    f"""
                    <div class="result-container1">
                        <span style="color: red; font-weight: bold; font-size: 1rem;">🚨 {result} Email Detected</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with col2:
    if 'confidence_non_fraudulent' in st.session_state and 'confidence_fraudulent' in st.session_state:
        confidence_non_fraudulent = st.session_state['confidence_non_fraudulent']
        confidence_fraudulent = st.session_state['confidence_fraudulent']

        # Add the heading for Confidence Scores
        st.markdown(
            """
            <div class="result-container">
                <p style="font-size: 2rem; font-weight: bold; color: white;">Confidence Scores:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display confidence percentages
        st.markdown(
            f"""
            <div class="percentage">
                <p style="font-size: 1.3rem;">Non-Fraudulent: <span style="font-size: 1.3rem;">{confidence_non_fraudulent:.2%}</span></p>
                <p style="font-size: 1.3rem;">Fraudulent: <span style="font-size: 1.3rem;">{confidence_fraudulent:.2%}</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Pie chart data
        labels = ['Non-Fraudulent', 'Fraudulent']
        sizes = [confidence_non_fraudulent, confidence_fraudulent]
        colors = ['#66D280', '#CC3950']
        explode = (0.1, 0)

        # # Create and display the pie chart
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.pie(
        #     sizes, explode=explode, labels=labels, colors=colors,
        #     autopct='%1.1f%%', shadow=False, startangle=140,
        #     textprops={'fontsize': 12, 'color': 'white'}
        # )
        # ax.axis('equal')
        # st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(50,50))  # Adjust size
        ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=140,
            textprops={'fontsize': 140, 'color': 'white'}
        )
        ax.axis('equal')  # Ensure pie is a circle

        # Transparent background for the chart
        fig.patch.set_facecolor('none')

        # Display the chart
        st.pyplot(fig)