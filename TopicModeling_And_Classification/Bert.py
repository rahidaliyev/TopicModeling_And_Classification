import os
from docx import Document
from sentence_transformers import SentenceTransformer, util
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from ReadDoc import read_doc
import win32com.client

# 2. Tokenləşdirmə (fraqmentlərə ayırmaq)
def get_candidate_keywords(text, ngram_range=(1, 3), stop_words='english'):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit([text])
    return vectorizer.get_feature_names_out()

# 3. Açar söz çıxarma (BERT + cosine similarity)
def extract_keywords(text, top_n=10):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    # Əsas mətni vektor halına sal
    doc_embedding = model.encode([text])[0]

    # Açar söz namizədlərini al
    candidates = get_candidate_keywords(text)
    candidate_embeddings = model.encode(candidates)

    # Uyğunluğu hesabla
    similarities = util.cos_sim(doc_embedding, candidate_embeddings)[0]
    
    # Ən yaxın olanları seç
    top_keywords = [candidates[i] for i in similarities.argsort(descending=True)[:top_n]]
    return top_keywords

# === Test ===
# file_path = r"C:\\Users\\rahid\\Documents\\projects\\advanced_machine_learning\\BERTZ\\text_files\\Alcohol.doc"
file_paths = [
    r"C:\\Users\\rahid\\Documents\\projects\\advanced_machine_learning\\BERTZ\\text_files\\Alcohol.doc",
    r"C:\\Users\\rahid\\Documents\\projects\\advanced_machine_learning\\BERTZ\\text_files\\DVI.doc",
    r"C:\\Users\\rahid\\Documents\\projects\\advanced_machine_learning\\BERTZ\\text_files\\Forensic_Paternity.doc"
]

#label-ləri yığmaq üçün kod
def extract_headings_doc(file_path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(file_path)
    headings = []

    for paragraph in doc.Paragraphs:
        text = paragraph.Range.Text.strip()
        if text.isupper() and len(text.split()) < 10:
            headings.append(text)
    doc.Close()
    word.Quit()
    return headings

#feature ve labellerin append edilmesi
keywords_list = []
labels = []
for file_path in file_paths:
    text = read_doc(file_path)
    keywords = extract_keywords(text)
    keywords_list.append(keywords)
    
    headings = extract_headings_doc(file_path)
    labels.extend(headings)
    
    print(f"\n Fayl: {os.path.basename(file_path)}")
    print(f"Fatures: {keywords}")
    print("Label-lar):")
    for heading in headings:
        print(" -", heading)
    
# Classifier 
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
keyword_vectors = [model.encode(keywords) for keywords in keywords_list]
keyword_vectors_combined = [np.mean(keywords, axis=0) for keywords in keyword_vectors]

#Building a Keyword-Based Classifier


# Fayl etiketləri (məsələn, Alcohol, DVI, Forensic Paternity)
# labels = ['Alcohol', 'DVI', 'Forensic_Paternity']

# BERT ilə əldə edilmiş açar söz vektorları
X = np.array(keyword_vectors_combined)

# 🔄 Normallaşdırma
scaler = Normalizer()
X_scaled = scaler.fit_transform(X)

# Train-test bölməsi
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Model qurmaq Logistic Regression
model = LogisticRegression(max_iter=1000)  # BERT embedding-lər üçün max_iter artırmaq faydalı olur
model.fit(X_train, y_train)

# Proqnozlaşdırma
y_pred = model.predict(X_test)

# Nəticə ölçülməsi
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")