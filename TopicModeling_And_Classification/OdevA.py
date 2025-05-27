import os
import docx
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch


folder_path = r'C:\Users\rahid\Documents\projects\advanced_machine_learning\TopicModeling_And_Classification\TopicModeling_And_Classification\text_files'

# 1. Extract observations (one observation per title)
def extract_observations_from_docx(file_path):
    doc = docx.Document(file_path)
    observations = []
    current_text = ""
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            if current_text:
                observations.append(current_text.strip())
                current_text = ""
        current_text += para.text + " "
    if current_text:
        observations.append(current_text.strip())
    return observations

# Remove observations from all documents
documents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".docx"):
        file_path = os.path.join(folder_path, filename)
        docs = extract_observations_from_docx(file_path)
        documents.extend(docs)

print(f"Observation count: {len(documents)}")

# 2. TF-IDF Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = tfidf_vectorizer.fit_transform(documents)

# 3. LDA Topic Modeling
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
X_lda = lda_model.fit_transform(X_tfidf)

# 4. BERT Embeddings
def get_bert_embeddings(text_list):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            mean_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(mean_embedding)
    return np.array(embeddings)

bert_embeddings = get_bert_embeddings(documents)

# 5. Combine all features
X_features = np.hstack([X_tfidf.toarray(), X_lda, bert_embeddings])

# 6. Remove useless features with Variance Threshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X_features)

print(f"Final shape (observation, feature): {X_selected.shape}")

# 7. Define a cluster for use as a "label" with KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_selected)

# 8. Normalization with StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# 9. Train-test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, clusters, test_size=0.2, random_state=42)

# 10. Logistic Regression model
model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# 11. Forecast and assessment
y_pred = model.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))