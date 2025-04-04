import mlflow
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pickle
import re
import kagglehub

# Download latest version
path = kagglehub.dataset_download("basilb2s/language-detection")

# Start MLflow tracking
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("language-detection")

# Load your data
df = pd.read_csv(path+'/Language Detection.csv')
X = df['Text']
y = df['Language']

# Preprocess data
data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with MLflow tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "MultinomialNB")
    mlflow.log_param("vectorizer", "CountVectorizer")
    
    # Create a proper scikit-learn Pipeline
    cv = CountVectorizer()
    model = MultinomialNB()
    
    # Build and train the pipeline
    pipe = Pipeline([('vectorizer', cv), ('model', model)])
    pipe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(pipe, "model")
    
    # Save pipeline for deployment
    with open("app/model/trained_pipeline-0.1.0.pkl", "wb") as f:
        pickle.dump(pipe, f)
    
    print(f"Model trained with accuracy: {accuracy}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")