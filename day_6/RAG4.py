

# Point wise Ranking using Logistic Regression 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Example dataset
data = {
    "Similarity Score": [0.85, 0.30, 0.75, 0.90, 0.45, 0.65, 0.80, 0.35, 0.78, 0.60],
    "Price":             [15,   50,   40,   20,   30,   55,   25,   60,   45,   35],
    "Rating":            [4.5,  3.0,  4.2,  4.8,  3.5,  4.0,  3.9,  3.1,  4.7,  3.9],
    "Label":             [1,    0,    1,    1,    0,    1,    0,    0,    1,    0]   
}

df = pd.DataFrame(data)

# Features and label
X = df[["Similarity Score", "Price", "Rating"]]
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression model
model = LogisticRegression()

# Fit model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   

# Print metrics
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))
print("Probability Scores for Class 1:", y_prob)




#  List-Wise Ranking 
from xgboost import XGBRanker
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Example data
documents = [
    "Artificial intelligence is transforming industries.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing helps computers using human language.",
    "FAISS is a library for efficient similarity search.",
    "Graph-based nearest-neighbor search is used in HNSW."
]

query = "What is Natural Language Processing?"


Y = [2, 1, 2, 0, 0]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Encode labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

groups = [len(documents)]

# Step 2: Train Ranking Model (XGBRanker)
model = XGBRanker(
    objective='rank:pairwise',
    learning_rate=0.1,
    n_estimators=100
)

model.fit(X, Y, group=groups)

# Step 3: Prediction (ranking scores)
preds = model.predict(X)

# Step 4: Sort documents by ranking score (descending)
ranked_docs = sorted(zip(preds, documents), reverse=True)

print("\nRanking Results:\n")
for score, doc in ranked_docs:
    print(f"Rank Score: {score:.4f}  →  {doc}")
