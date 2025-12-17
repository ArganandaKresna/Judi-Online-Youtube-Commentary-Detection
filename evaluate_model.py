import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs("evaluation_results", exist_ok=True)

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv('labeled_kandidat_spam.csv', sep=';')
    df['label'] = df['label'].astype(int)
    df['comment_text'] = df['comment_text'].astype(str)
    
    # Same split as training to ensure we test on unseen data
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Test Set Size: {len(test_df)}")

except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# 2. Load Model
print("Loading Model...")
MODEL_PATH = "./indobert-spam-detection-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 3. Prediction Loop
predictions = []
true_labels = []
confidences = []

print("Running predictions...")
for index, row in test_df.iterrows():
    text = row['comment_text']
    label = row['label']
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()
        
    predictions.append(pred_idx)
    true_labels.append(label)
    confidences.append(confidence)

# 4. Evaluation Metrics
acc = accuracy_score(true_labels, predictions)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=["Aman", "Spam"]))

# 5. Visualizations

# A. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Aman", "Spam"], yticklabels=["Aman", "Spam"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('evaluation_results/confusion_matrix.png')
print("Saved confusion_matrix.png")

# B. Confidence Distribution
plt.figure(figsize=(10, 6))
sns.histplot(confidences, bins=20, kde=True, color='green')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.title('Prediction Confidence Distribution')
plt.savefig('evaluation_results/confidence_distribution.png')
print("Saved confidence_distribution.png")

print("\nEvaluation Complete. Check 'evaluation_results' folder for graphs.")
