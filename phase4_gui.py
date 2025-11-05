# ==============================================
# Phase 4: GUI for Spam Detection
# ==============================================

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import re

# ---------------------------------------------------------
# Load model and preprocessing
# ---------------------------------------------------------
try:
    data = joblib.load("preprocessed_data_full.joblib")
    vectorizer = data["vectorizer"]
    stopwords = data["stopwords"]
    model = joblib.load("logistic_regression_model.joblib")
except Exception as e:
    messagebox.showerror("Error", f"Missing model/data files: {e}")
    raise SystemExit

# ---------------------------------------------------------
# Text cleaning function
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join([w for w in text.split() if w not in stopwords and len(w) > 1])
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_spam():
    raw_text = text_input.get("1.0", tk.END).strip()
    if not raw_text:
        messagebox.showwarning("Warning", "Please enter an email message.")
        return

    cleaned = clean_text(raw_text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]

    label_text = "SPAM (1)" if pred == 1 else "HAM (0)"
    label_color = "red" if pred == 1 else "green"

    result_label.config(
        text=f"Prediction: {label_text}",
        foreground=label_color,
        font=("Segoe UI", 14, "bold")
    )

# ---------------------------------------------------------
# GUI Layout
# ---------------------------------------------------------
root = tk.Tk()
root.title("Enron Spam Detector")
root.geometry("600x400")
root.resizable(False, False)

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
style.configure("TLabel", font=("Segoe UI", 11))
style.configure("TFrame", background="#FFFFFF")

frame = ttk.Frame(root, padding=20)
frame.pack(fill="both", expand=True)

ttk.Label(frame, text="Enter email text to classify:", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 5))
text_input = tk.Text(frame, height=10, wrap="word", font=("Segoe UI", 10))
text_input.pack(fill="x", pady=5)

predict_button = ttk.Button(frame, text="Detect Spam", command=predict_spam)
predict_button.pack(pady=10)

result_label = ttk.Label(frame, text="Prediction: (none)", font=("Segoe UI", 12))
result_label.pack(pady=10)

ttk.Label(frame, text="Model: Logistic Regression | TF-IDF | Enron Dataset", font=("Segoe UI", 9, "italic")).pack(side="bottom", pady=(10, 0))

root.mainloop()
