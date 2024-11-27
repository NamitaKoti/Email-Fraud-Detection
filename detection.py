import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved BERT model and tokenizer
model_path = r'bert_model'  # Ensure the correct path
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Preprocess the text for BERT
def transform_text(text):
    encodings = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,  # Adjust this based on your training parameters
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']

# Streamlit app configuration
st.set_page_config(page_title="Email Fraud Detection", layout="centered")

# Adding custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.pinimg.com/736x/6e/46/da/6e46da2c1712b7daaba49f78988221a4.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.9);
        color: black;
    }
    .stButton>button {
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
    .result-box {
        margin-top: 1rem;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("📧 Email Fraud Detection")
st.write("Detect whether an email is Fraudulent or Non-fraudulent.")

# Text input for the email content
input_sms = st.text_area("Enter the email text below:")

# Button to analyze email
if st.button('Analyze Email'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter text before analyzing.")
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

            # Determine result based on threshold
            threshold = 0.5
            result = "Fraudulent" if confidence_fraudulent >= threshold else "Non-Fraudulent"

            # Display result in a styled box
            if result == "Fraudulent":
                st.markdown(
                    f"<div class='result-box' style='color: red;'>🚨 {result} Email Detected</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-box' style='color: green;'>✅ {result} Email Detected</div>",
                    unsafe_allow_html=True
                )

            # Display confidence scores
            st.markdown(
                f"""
                <div class='result-box'>
                    <p>Non-Fraudulent: {confidence_non_fraudulent:.2%}</p>
                    <p>Fraudulent: {confidence_fraudulent:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
