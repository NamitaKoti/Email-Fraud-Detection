# -*- coding: utf-8 -*-
"""Success.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1M7vKQR4jH6FdvPOjLMJywPkL4RNkKxu3
"""

import numpy as np
import pandas as pd

import pandas as pd
from google.colab import files

# Upload the CSV file
uploaded = files.upload()

# List of possible encodings to try
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

# Attempt to read the CSV file with different encodings
for encoding in encodings:
    try:
        df = pd.read_csv('spam.csv', encoding=encoding)  # Use the uploaded filename
        print(f"File successfully read with encoding: {encoding}")
        print(df.head())  # Display the first few rows of the DataFrame
        break  # Stop the loop if successful
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")
        continue  # Try the next encoding
    except FileNotFoundError:
        print("File not found. Please upload the file.")
        break  # Exit the loop if the file is not found

df.sample(5)

import random
from datetime import datetime, timedelta

# Define a function to generate random timestamps
def generate_random_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

# Define a start and end date for the timestamps
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)

# Add a new column with random timestamps
df['Timestamp'] = [generate_random_timestamp(start_date, end_date) for _ in range(len(df))]

# Display the first few rows to verify
print(df.head())

df.shape

# 1.Data Cleaning
# 2.EDA
# 3.Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvements
# 7. Website
# 8. Deploy

# 1.Data Cleaning
# 2.EDA
# 3.Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvements
# 7. Website
# 8. Deploy

"""1. DATA CLEANING"""

df.info()

# Drop last three columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

df.sample(5)

#renaming cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])

df.head()

# missing values
df.isnull().sum()

#check duplicate values
df.duplicated().sum()

# remove duplicates
df = df.drop_duplicates(keep='first')

df.duplicated().sum()

df.shape

"""2. EDA"""

df.head()

df['target'].value_counts() ## ham more than spam

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()

# Data is imbalanced

import nltk

!pip install nltk

nltk.download('punkt')

# length of each text
df['num_characters'] = df['text'].apply(len)

df.head()

# num of words
df['text'].apply(lambda x:nltk.word_tokenize(x))

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df.head()

# number of sentences
df['text'].apply(lambda x:nltk.sent_tokenize(x))

df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head()

df[['num_characters','num_words','num_sentences']].describe()

# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()

# spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()

import seaborn as sns

plt.figure(figsize=(10,5))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='black')

plt.figure(figsize=(10,5))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='black')

plt.figure(figsize=(10,5))
sns.histplot(df[df['target'] == 0]['num_sentences'])
sns.histplot(df[df['target'] == 1]['num_sentences'],color='black')

sns.pairplot(df,hue='target') # 0 - ham , 1 - spam

# Generate the heatmap for the correlation matrix of the specified columns directly
sns.heatmap(df[['target','num_characters', 'num_words', 'num_sentences',]].corr(), annot=True,fmt='.2f')

# Add title
plt.title('Correlation Heatmap of Selected Features')
plt.show()

print(df.columns)  # Check column names to verify the presence of 'Timestamp' and 'DayOfWeek'

# Convert 'Timestamp' to datetime format (if it's not already in that format)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')  # 'coerce' will handle any invalid dates

# Check if the conversion was successful
print(df['Timestamp'].head())

# Extract the day of the week (e.g., Monday, Tuesday, etc.)
df['DayOfWeek'] = df['Timestamp'].dt.day_name()

# Check the new column to ensure it's added correctly
print(df[['Timestamp', 'DayOfWeek']].head())

# Perform one-hot encoding for the DayOfWeek column (excluding the first column to avoid multicollinearity)
df = pd.get_dummies(df, columns=['DayOfWeek'], drop_first=True)

# Check the updated DataFrame to verify that one-hot encoding was applied correctly
print(df.head())

# Assuming 'DayOfWeek' columns are already created (from your timestamp column)
df['DayOfWeek'] = df['Timestamp'].dt.day_name()  # Extracting day of the week from Timestamp

# Plot the day-of-week distribution for spam and ham emails
plt.figure(figsize=(10, 6))
sns.countplot(x='DayOfWeek', hue='target', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Day of Week Distribution for Spam vs. Ham Emails')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Extract hour from the Timestamp
df['hour'] = df['Timestamp'].dt.hour

# Plot the distribution of emails based on the hour of the day for spam and ham
plt.figure(figsize=(8, 6))
sns.countplot(x='hour', hue='target', data=df)
plt.title('Email Distribution Based on Hour of Day (Spam vs. Ham)')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()

"""3. DATA PREPROCESSING"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords') # you may need to download the stopwords dataset

ps = PorterStemmer()

def transform_text(text):
    text = text.lower() # to convert into lower case
    text = nltk.word_tokenize(text) # tokenization

    y = []
    for i in text:
        if i.isalnum(): # if its alpha numeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    #Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

df.head()

from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=""))

plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=""))

plt.imshow(ham_wc)

df.head()

spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

len(spam_corpus)

from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')  # Corrected plt.xticks
plt.show()

ham_corpus = []
for msg in df[df['target']== 0 ]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

len(ham_corpus)

from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0], y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')  # Corrected plt.xticks
plt.show()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert to datetime

# Extract day of the week, hour of the day, day of the month, and month
df['day_of_week'] = df['Timestamp'].dt.day_name()  # Name of the day
df['hour'] = df['Timestamp'].dt.hour  # Hour of the day
df['day_of_month'] = df['Timestamp'].dt.day  # Day of the month
df['month'] = df['Timestamp'].dt.month  # Month number (1-12)

# Create is_weekend flag
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

# Check for missing values (if any)
df.isnull().sum()

# If needed, fill missing values (for numerical or categorical columns)
# df['column_name'].fillna(value, inplace=True)

# One-Hot Encoding for day_of_week (optional, depends on the model)
df = pd.get_dummies(df, columns=['day_of_week'], drop_first=True)

# Label encoding for month or other categorical features (if needed)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['month_encoded'] = encoder.fit_transform(df['month'])

# Drop the Timestamp column (if no longer needed) or any other redundant columns
df.drop(['Timestamp'], axis=1, inplace=True)

print(df.columns)

# Save the processed DataFrame to a new CSV file
df.to_csv('processed_email_data.csv', index=False)

from google.colab import files
files.download('processed_email_data.csv')

"""step1 : Feature Engineering"""

# Extracting the necessary features from your DataFrame
text_features = df[['num_characters', 'num_words', 'num_sentences'] + [col for col in df.columns if 'DayOfWeek' in col]].values

"""step2 : Tokenization and BERT Embedding"""

!pip install transformers

!pip install torch

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your dataset
# df = pd.read_csv('path_to_your_data.csv') # Load your data here

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
max_length = 128
encodings = tokenizer(df['transformed_text'].tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')

input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']

# Convert labels to tensor
labels = torch.tensor(df['target'].values)

# Add additional features (numeric data)
# Assuming you have columns like 'num_characters', 'num_words', 'num_sentences', 'is_weekend', etc.
numeric_features = df[['num_characters', 'num_words', 'num_sentences', 'is_weekend']].values
additional_features = torch.tensor(numeric_features, dtype=torch.float32)

# Ensure that input_ids, attention_masks, and additional features are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)
additional_features = additional_features.to(device)
labels = labels.to(device)

# Split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=0.2, random_state=42)
train_additional, val_additional, _, _ = train_test_split(additional_features, input_ids, test_size=0.2, random_state=42)

# Convert to torch Dataset
train_data = TensorDataset(train_inputs, train_masks, train_additional, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)

validation_data = TensorDataset(val_inputs, val_masks, val_additional, val_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=16)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Adjust num_labels based on your task
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Define the learning rate scheduler
epochs = 6  # Adjust based on your needs
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_dataloader):
        batch_input_ids, batch_attention_masks, batch_additional_features, batch_labels = [t.to(device) for t in batch]

        model.zero_grad()
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

# Evaluation loop
model.eval()  # Set the model to evaluation mode
predictions, true_labels = [], []

with torch.no_grad():
    for batch in validation_dataloader:
        batch_input_ids, batch_attention_masks, batch_additional_features, batch_labels = [t.to(device) for t in batch]

        # Pass through the model
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        logits = outputs.logits

        # Convert logits to predictions
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(batch_labels.tolist())

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

from transformers import BertForSequenceClassification, BertTokenizer
import torch
import shutil
from google.colab import files

# Save the trained model and tokenizer
save_directory = '/content/bert_model'  # Define the directory where the model and tokenizer will be saved
model.save_pretrained(save_directory)  # This saves the model and configuration files
tokenizer.save_pretrained(save_directory)  # This saves the tokenizer files

# Optionally, save the model's state_dict if you need it for manual loading later
torch.save(model.state_dict(), f'{save_directory}/bert_model_state_dict.pth')

# Print a confirmation message
print("Model and tokenizer have been saved successfully.")

# Step 1: Zip the saved model and tokenizer files into a single ZIP archive
zip_filename = '/content/bert_model_files.zip'
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', save_directory)

# Step 2: Provide a download link for the ZIP file
files.download(zip_filename)  # This will prompt the download of the ZIP file

print(f"The model and tokenizer files have been zipped and are available for download: {zip_filename}")