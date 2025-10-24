# ==============================================
# Phase 2: Model Training
# ==============================================

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")  # suppress minor warnings for clarity

# ---------------------------------------------------------
# Step 1: Load preprocessed data
# ---------------------------------------------------------
print("📦 Loading preprocessed data...")
data = joblib.load("preprocessed_data_full.joblib")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
print(f"Classes in training set: {y_train.value_counts().to_dict()}")

# ---------------------------------------------------------
# Step 2: Define models
# ---------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# ---------------------------------------------------------
# Step 3: Train, evaluate, and save each model
# ---------------------------------------------------------
for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Results:")
        print(f"  Accuracy : {acc:.4f}")
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Save model
        joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.joblib")
        print(f"💾 Saved {name} model to '{name.replace(' ', '_').lower()}_model.joblib'")

    except ValueError as e:
        print(f"⚠️ {name} could not be trained: {e}")

print("\n✅ All models processed successfully!")
