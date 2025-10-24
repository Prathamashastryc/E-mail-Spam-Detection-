import subprocess
import nltk
from pathlib import Path

# === Step 1: Define all dependencies ===
packages = [
    "pandas",
    "numpy",
    "scikit-learn",
    "nltk",
    "spacy",
    "matplotlib",
    "seaborn",
    "wordcloud",
    "streamlit",
    "transformers",
    "tqdm",
    "joblib"
]

print("📦 Installing required Python packages...\n")
subprocess.check_call(["pip", "install", "--upgrade", "pip"])
subprocess.check_call(["pip", "install", *packages])

# === Step 2: Download NLTK data ===
print("\n📚 Downloading NLTK resources...")
nltk_data_dir = Path.home() / "AppData" / "Roaming" / "nltk_data"
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)

# === Step 3: Download SpaCy English model ===
print("\n🧠 Downloading SpaCy English model (en_core_web_sm)...")
subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])

# === Step 4: Confirm setup ===
print("\n✅ Environment setup complete!")
print("You can now run: python phase1_preprocessing.py")
