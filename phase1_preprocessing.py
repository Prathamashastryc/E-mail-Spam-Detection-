# ==============================================
# Phase 1: Preprocessing (Enron Spam Filter)
# ==============================================

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import joblib
import os

# ---------------------------------------------------------
# Step 1: NLTK Setup
# ---------------------------------------------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------
# Step 2: Load the Enron dataset (local file)
# ---------------------------------------------------------
file_path = "emails.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(
        "❌ Dataset not found! Please ensure 'emails.csv' is in the same folder as this script."
    )

df = pd.read_csv(file_path, encoding="latin1")
print(f"📥 Loaded {len(df)} emails")

# ---------------------------------------------------------
# Step 3: Normalize column names
# ---------------------------------------------------------
# Different versions of the dataset use different headers.
if "message" in df.columns:
    df.rename(columns={"message": "text"}, inplace=True)
elif "content" in df.columns:
    df.rename(columns={"content": "text"}, inplace=True)
elif "Message" in df.columns:
    df.rename(columns={"Message": "text"}, inplace=True)

# Add a label column if it exists
if "spam" in df.columns:
    df["label_num"] = df["spam"].apply(lambda x: 1 if x in [1, "spam", True] else 0)
elif "label" in df.columns:
    df["label_num"] = df["label"].apply(lambda x: 1 if str(x).lower() == "spam" else 0)
else:
    # Dummy labels if the dataset has none — can update later
    df["label_num"] = 0

# Keep only text and label columns
df = df[["text", "label_num"]].dropna()

# ---------------------------------------------------------
# Step 4: Clean the email text
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"<.*?>", "", text)           # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

tqdm.pandas(desc="🧹 Cleaning emails")
df["clean_text"] = df["text"].progress_apply(clean_text)

# ---------------------------------------------------------
# Step 5: Split data for training/testing
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label_num"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"] if df["label_num"].nunique() > 1 else None,
)

print(f"📊 Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# ---------------------------------------------------------
# Step 6: TF-IDF Vectorization
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("✅ Data cleaned and vectorized successfully!")

# ---------------------------------------------------------
# Step 7: Save preprocessed data
# ---------------------------------------------------------
joblib.dump(
    (X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer),
    "preprocessed_data.joblib"
)
print("💾 Saved preprocessed data to preprocessed_data.joblib")
