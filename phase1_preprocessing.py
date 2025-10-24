# ==============================================
# Phase 1: Preprocessing (Robust Version)
# ==============================================

import os
import re
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import joblib

# ---------------------------------------------------------
# Step 1: Setup NLTK stopwords
# ---------------------------------------------------------
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------
# Step 2: Load dataset
# ---------------------------------------------------------
local_main = "emails.csv"
local_fallback = "enron_spam_data.csv"
df = None

if os.path.exists(local_main):
    print(f"📥 Found local dataset: {local_main}")
    df = pd.read_csv(local_main, encoding="latin1")
elif os.path.exists(local_fallback):
    print(f"📥 Using fallback dataset: {local_fallback}")
    df = pd.read_csv(local_fallback, encoding="latin1")
else:
    # Try downloading from mirrors
    urls = [
        "https://huggingface.co/datasets/bvk/ENRON-spam/resolve/main/enron_spam_data.csv",
        "https://raw.githubusercontent.com/bdanalytics/Enron-Spam/master/data/enron_spam_data.csv",
    ]
    for url in urls:
        try:
            print(f"🌐 Downloading dataset from: {url}")
            df = pd.read_csv(url, encoding="latin1")
            print("✅ Download successful")
            break
        except Exception as e:
            print(f"⚠️ Failed to download: {e}")

if df is None:
    raise FileNotFoundError("❌ Could not load any labeled dataset. Place 'emails.csv' or 'enron_spam_data.csv' here.")

# ---------------------------------------------------------
# Step 3: Standardize columns
# ---------------------------------------------------------
# Detect text column
text_cols = [c for c in df.columns if c.lower() in ["text", "message", "emailtext", "content", "body", "subject"]]
df["text"] = df[text_cols[0]].astype(str) if text_cols else df.iloc[:, 0].astype(str)

# Detect label column
label_cols = [c for c in df.columns if c.lower() in ["label", "spam", "target", "category", "spam/ham"]]
if label_cols:
    df["label"] = df[label_cols[0]].astype(str)
else:
    raise ValueError("❌ Could not detect a label column.")

# ---------------------------------------------------------
# Step 4: Robust label mapping
# ---------------------------------------------------------
def map_label(x):
    x = str(x).strip().lower()
    if x in ["spam", "1", "true", "yes"]:
        return 1
    elif x in ["ham", "0", "false", "no"]:
        return 0
    else:
        return None  # ignore unknown labels

df["label_num"] = df["label"].apply(map_label)
df = df.dropna(subset=["label_num"])
df["label_num"] = df["label_num"].astype(int)

# Ensure both classes exist
if df["label_num"].nunique() < 2:
    raise ValueError("❌ Dataset contains only one class. Check the label mapping.")

print(f"📊 Dataset loaded: {len(df)} emails | Spam: {df['label_num'].sum()} | Ham: {len(df) - df['label_num'].sum()}")

# ---------------------------------------------------------
# Step 5: Clean text
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"<.*?>", "", text)               # remove HTML
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = " ".join([w for w in text.split() if w not in STOPWORDS and len(w) > 1])
    text = re.sub(r"\s+", " ", text).strip()
    return text

tqdm.pandas(desc="🧹 Cleaning emails")
df["clean_text"] = df["text"].progress_apply(clean_text)

# ---------------------------------------------------------
# Step 6: Split data
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label_num"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"]
)

print(f"📚 Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(y_train.value_counts(), y_test.value_counts())

# ---------------------------------------------------------
# Step 7: Vectorize
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("✅ Data cleaned and vectorized successfully!")

# ---------------------------------------------------------
# Step 8: Save preprocessed data
# ---------------------------------------------------------
joblib.dump({
    "X_train": X_train_tfidf,
    "X_test": X_test_tfidf,
    "y_train": y_train,
    "y_test": y_test,
    "vectorizer": vectorizer,
    "stopwords": STOPWORDS
}, "preprocessed_data_full.joblib")

print("💾 Saved preprocessed data to 'preprocessed_data_full.joblib'")
