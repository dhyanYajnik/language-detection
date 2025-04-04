import pickle
import re
from pathlib import Path

__version__ = '0.1.0'

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the pipeline
with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    pipe = pickle.load(f)
    
classes = [
    'Arabic', 'Danish', 
    'Dutch', 'English',
    'French', 'German',
    'Greek', 'Hindi', 
    'Italian', 'Kannada', 
    'Malayalam', 'Portugeese',
    'Russian', 'Spanish', 
    'Sweedish', 'Tamil', 
    'Turkish'
]

def predict_pipeline(text):
    # Preprocess the text
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    
    # Check if pipe is a dictionary (as saved in train_with_mlflow.py)
    # or a sklearn Pipeline object (as in the notebook)
    if isinstance(pipe, dict):
        # If it's a dictionary, use the components directly
        vectorized = pipe["vectorizer"].transform([text])
        pred = pipe["model"].predict(vectorized)
    else:
        # If it's a sklearn Pipeline, use it directly
        pred = pipe.predict([text])
    
    return classes[pred[0]]