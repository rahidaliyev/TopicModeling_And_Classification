# Lazımi kitabxanalar
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
import win32com.client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from ReadDoc import read_doc

# Fayl yolları və label-lar
file_paths = [
    r"C:\Users\rahid\Documents\projects\advanced_machine_learning\TopicModeling_And_Classification\TopicModeling_And_Classification\text_files\Alcohol.doc",
    r"C:\Users\rahid\Documents\projects\advanced_machine_learning\TopicModeling_And_Classification\TopicModeling_And_Classification\text_files\DVI.doc",
    r"C:\Users\rahid\Documents\projects\advanced_machine_learning\TopicModeling_And_Classification\TopicModeling_And_Classification\text_files\Forensic_Paternity.doc"
]
labels_map = ["alcohol", "dvi", "forensic"]

# ================================
# Faylları oxu və data yığ
all_texts = []
all_labels = []

for path, label in zip(file_paths, labels_map):
    full_text = read_doc(path)
    paragraphs = full_text.split('\r')  # Paraqraflara böl
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    all_texts.extend(paragraphs)
    all_labels.extend([label] * len(paragraphs))

# DataFrame
df = pd.DataFrame({"text": all_texts, "label": all_labels})
print("Total doc count:", len(df))

# ================================
# TF-IDF çıxarımı
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df["text"])

# ================================
# LDA topic modeling
lda = LatentDirichletAllocation(n_components=3, random_state=42)
X_lda = lda.fit_transform(X_tfidf)

# ================================
# BERT embeddings
model_bert = SentenceTransformer('all-MiniLM-L6-v2')
X_bert = model_bert.encode(df["text"], show_progress_bar=True)

# ================================
# Klassifikasiya funksiyası
def train_and_evaluate_LR(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n==== {title} for results ====")
    print(classification_report(y_test, y_pred))
    
def train_and_evaluate_SVM(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', random_state=42)  # 'linear' kernel seçilib
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n==== {title} for results ====")
    print(classification_report(y_test, y_pred))

# ================================
# Nəticələri yoxla
train_and_evaluate_LR(X_tfidf, df["label"], "TF-IDF with Linear Regression")
train_and_evaluate_LR(X_lda, df["label"], "LDA with Linear Regression")
train_and_evaluate_LR(X_bert, df["label"], "BERT with Linear Regression")

# Nəticələri yoxla
train_and_evaluate_SVM(X_tfidf, df["label"], "TF-IDF with SVM")
train_and_evaluate_SVM(X_lda, df["label"], "LDA with SVM")
train_and_evaluate_SVM(X_bert, df["label"], "BERT with SVM")





def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx}: ", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-top_n:]])

print_topics(lda, vectorizer)
