# ==============================================
# Phase 3: Inference Script
# ==============================================

import joblib
import re
import pandas as pd

# ---------------------------------------------------------
# Step 1: Load preprocessed objects and model
# ---------------------------------------------------------
data = joblib.load("preprocessed_data_full.joblib")
vectorizer = data["vectorizer"]
stopwords = data["stopwords"]

# Load the model you want to use
# Options: naive_bayes_model.joblib, logistic_regression_model.joblib, random_forest_model.joblib
model_path = "logistic_regression_model.joblib"
model = joblib.load(model_path)

print(f"✅ Loaded model from '{model_path}'")

# ---------------------------------------------------------
# Step 2: Define text cleaning function
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"<.*?>", "", text)               # remove HTML
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = " ".join([w for w in text.split() if w not in stopwords and len(w) > 1])
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------
# Step 3: Prepare raw emails for inference
# ---------------------------------------------------------
# Example: list of emails to classify
raw_emails = [
    "Congratulations! You won a free iPhone. Click here to claim.",
    "Hi team, please find the attached report for last week.",
    "Limited time offer! Get 50% off on your subscription."
]

# Clean emails
cleaned_emails = [clean_text(email) for email in raw_emails]

# Vectorize
X_input = vectorizer.transform(cleaned_emails)

# ---------------------------------------------------------
# Step 4: Predict
# ---------------------------------------------------------
predictions = model.predict(X_input)
pred_labels = ["Spam" if p == 1 else "Ham" for p in predictions]

# ---------------------------------------------------------
# Step 5: Display results
# ---------------------------------------------------------
results = pd.DataFrame({
    "Email": raw_emails,
    "Prediction": pred_labels
})

print("\n📌 Inference Results:")
print(results)
