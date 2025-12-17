import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create output directory
os.makedirs("evaluation_results", exist_ok=True)

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv('labeled_kandidat_spam.csv', sep=';')
    df['label'] = df['label'].astype(int)
    df['comment_text'] = df['comment_text'].astype(str)
    
    # Same split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Test Set Size: {len(test_df)}")

except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# 2. Load Model (We need the base BERT model to get embeddings)
print("Loading Model...")
MODEL_PATH = "./indobert-spam-detection-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# Load as BertModel to easily access hidden states
model = BertModel.from_pretrained(MODEL_PATH)
model.eval()

# 3. Extract Embeddings
embeddings = []
labels = []

print("Extracting embeddings...")
for index, row in test_df.iterrows():
    text = row['comment_text']
    label = row['label']
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the CLS token embedding (first token)
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
    embeddings.append(cls_embedding[0])
    labels.append(label)

embeddings = np.array(embeddings)
labels = np.array(labels)

# 4. Dimensionality Reduction (PCA to 3D)
print("Performing PCA...")
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

# 5. 3D Plotting
print("Generating 3D Plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors and markers
colors = ['blue', 'red']
class_names = ['Aman', 'Spam']

for class_idx in [0, 1]:
    # Select points for this class
    mask = labels == class_idx
    ax.scatter(
        reduced_embeddings[mask, 0], 
        reduced_embeddings[mask, 1], 
        reduced_embeddings[mask, 2], 
        c=colors[class_idx], 
        label=class_names[class_idx],
        s=50,
        alpha=0.7
    )

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D Visualization of BERT Embeddings (Test Set)')
ax.legend()

output_path = 'evaluation_results/3d_embeddings.png'
plt.savefig(output_path)
print(f"Saved 3D graph to {output_path}")
