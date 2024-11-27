import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved BERT model and tokenizer (this will point to the model directory after downloading)
model_path = r'bert_model'  # Corrected path with forward slashes
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
    /* Set the background image for the app */
    *{
        box-sizing: border-box
    }
    .stApp {
        background-image: url("https://i.pinimg.com/736x/6e/46/da/6e46da2c1712b7daaba49f78988221a4.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white; /* Default text color */
    }

    /* Custom title styling */
    .custom-title {
        float: left;
        margin-left: -24rem;
        font-weight: 1000;
        font-size: 3rem;
        font-family: lucida;
    }

    /* Styling for the left input section */
    .left-section {
        float: left;
        margin-left: -24rem;
        font-weight: 500;
        font-size: 1.7rem;
        font-family: lucida;
    }

    /* Text area styling */
   .stTextArea {
        margin-left: -24rem !important;
        width: 600px !important; /* Full width */
        # resize: none; /* Disable resizing */
    }

    /* Hover effect for the text area */
    .stTextArea textarea:hover {
        transition: font-size 0.3s ease;
        # resize: none !important;
    }

    # /* Button styling */
    # .stButton>button {
    #     margin-left: -24rem !important;
    #     background-color: #008CBA; /* Blue background */
    #     color: white;
    #     padding: 10px 20px;
    #     font-size: 16px;
    #     border-radius: 5px; /* Rounded corners */
    #     border: none;
    #     cursor: pointer;
    #     transition: background-color 0.3s ease;
    # }

    .stButton>button {
    margin-left: -24rem !important;
    background-color: #008CBA; /* Blue background */
    color: white; /* Default text color */
    padding: 10px 20px;
    font-size: 16px;
    border-radius: 5px; /* Rounded corners */
    border: none;
    cursor: pointer;
    transition: color 0.3s ease, background-color 0.3s ease; /* Smooth transition for text color and background */
}

.stButton>button:hover {
    color: white; /* Text color on hover */
    background-color: #005f73; /* Darker shade of blue when hovered */
}

.stButton>button:active {
    color: white; /* Ensure text color stays white when the button is pressed */
    background-color: #007ea7; /* Keep a consistent background color when the button is clicked */
}


    /* Alert box styling */
    .custom-alert {
        margin-top: 20px;
        background-color: #f8d7da; /* Light red background */
        color: #721c24; /* Dark red text */
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Container for results and confidence scores */
    .result-container1{
        float: left;
        margin-left: -24rem;
        # margin-right: -50rem;
        background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent background */
        border-radius: 10px;
        padding: 20px;
        margin-top: 1rem;
        width: 100%; /* Adjust width */
        max-width: 600px;
        font-size: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .result-container {
        float: left;
        # margin-right: -50rem;
        # background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent background */
        # border-radius: 10px;
        # padding: 5px;
        # margin-top: 2rem;
        # width: 100%; /* Adjust width */
    #     box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    font-size: large;
    }

    /* Pie chart styling without background */
.pie-chart {
        margin-top: 2rem;
        margin-right: -10rem;
        background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent background */
        border-radius: 10px;
        padding: 20px;
        margin-top: 2rem;
        width: 80%;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .percentage {
          float: left;
        # margin-right: -50rem;
        background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent background */
        border-radius: 10px;
        padding: 20px;
        margin-top: -1rem;
        width: 100%; /* Adjust width */
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

                # Determine result based on threshold
                threshold = 0.5  # Adjust this as needed
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
    if input_sms.strip() != "":  # Display only when there's input
        st.markdown(
            f"""
            <div class="result-container">
                <p style="font-size: 2rem; font-weight: bold;">Confidence Scores:</p>
            </div>
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
        colors = ['#66D280' , '#CC3950']  # Green for non-fraudulent, red for fraudulent
        explode = (0.1, 0)  # Highlight non-fraudulent section


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