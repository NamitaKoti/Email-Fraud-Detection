import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved BERT model and tokenizer (this will point to the model directory after downloading)
model_path = r'C:/Success-4-1/bert_model'  # Corrected path with forward slashes
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Preprocess the text for BERT
def transform_text(text):
    # Tokenize and encode the input
    encodings = tokenizer(       text,        truncation=True,
        padding='max_length',
        max_length=128,  # Adjust this based on your training parameters
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']

# Streamlit app
st.set_page_config(page_title="Email Fraud Detection", layout="centered")
st.title("📧 Email Fraud Detection")

st.write("Detect whether an email is Fraudulent or Non-fraudulent.")
input_sms = st.text_area("Enter the email text below:")

if st.button('Analyze Email'):
    if input_sms.strip() == "":
        st.warning("Please enter text before analyzing.")
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
            threshold = 0.5  # Adjust this as needed
            result = "Fraudulent" if confidence_fraudulent >= threshold else "Non-Fraudulent"

            # Display the result with confidence
            if result == "Fraudulent":
                st.error(f"🚨 {result} Email Detected")
            else:
                st.success(f"✅ {result} Email Detected")

            # Display confidence
            st.write("### Confidence Scores:")
            st.write(f"- Non-Fraudulent: {confidence_non_fraudulent:.2%}")
            st.write(f"- Fraudulent: {confidence_fraudulent:.2%}")