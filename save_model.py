from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Save the model's state_dict (weights) to a file
model.save_pretrained('./bert_model')

# Load the tokenizer for the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Save the tokenizer
tokenizer.save_pretrained('./bert_model')

print("Model and tokenizer saved successfully!")