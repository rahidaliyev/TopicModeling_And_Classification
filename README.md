# Topic Modeling and Classification

This project demonstrates topic modeling and classification using three different feature extraction techniques: **TF-IDF**, **LDA**, and **BERT embeddings**. The extracted features are used to train and evaluate classifiers such as **Logistic Regression** and **SVM**.

## Project Structure

### Files and Directories

- **`Main.py`**: The main script that performs feature extraction, topic modeling, and classification.
- **`Bert.py`**: Contains additional functionality for keyword extraction and clustering using BERT embeddings.
- **`ReadDoc.py`**: A utility script to read `.doc` files using the `win32com.client` library.
- **`text_files/`**: Contains the `.doc` files used for training and testing.
- **`.vscode/settings.json`**: Configuration for the Python environment in Visual Studio Code.

## Requirements

To run this project, you need the following Python libraries:

- `pandas`
- `scikit-learn`
- `sentence-transformers`
- `numpy`
- `win32com.client`
- `nltk`
- `torch`
- `transformers`

Install the required libraries using:

```bash
pip install pandas scikit-learn sentence-transformers numpy pywin32 nltk torch transformers
How to Run
Place your .doc files in the text_files/ directory.
Run the Main.py script to perform the following tasks:
Read the .doc files and preprocess the text.
Extract features using TF-IDF, LDA, and BERT embeddings.
Train and evaluate classifiers (Logistic Regression and SVM) using the extracted features.
Print the classification results and LDA topics.

python [Main.py](http://_vscodecontentref_/2)
Features
1. Feature Extraction
TF-IDF: Extracts term frequency-inverse document frequency features.
LDA: Performs Latent Dirichlet Allocation for topic modeling.
BERT: Generates sentence embeddings using the all-MiniLM-L6-v2 model.
2. Classification
Logistic Regression: A linear model for classification.
SVM: Support Vector Machine with a linear kernel.
3. Topic Modeling
Extracts and prints the top words for each topic using LDA.
Results
The script evaluates the performance of each feature extraction method with both classifiers and prints metrics such as precision, recall, and F1-score.
Example Output
==== TF-IDF with Linear Regression for results ====
              precision    recall  f1-score   support
...

==== LDA with SVM for results ====
              precision    recall  f1-score   support
...

Topic 0: ['word1', 'word2', 'word3', ...]
Topic 1: ['word4', 'word5', 'word6', ...]
Notes
Ensure that Microsoft Word is installed on your system for the win32com.client library to work.
The Bert.py script provides additional functionality for keyword extraction and clustering but is not directly used in Main.py.
Future Improvements
Add hyperparameter tuning for classifiers.
Experiment with additional feature extraction techniques.
Use a larger dataset for better generalization.
License
This project is for educational purposes only.
