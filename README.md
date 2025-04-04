Language Detection API
======================

A machine learning application that detects the language of input text. The system supports 17 languages including Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portuguese, Russian, Spanish, Swedish, Tamil, and Turkish.

Project Architecture
--------------------

This project implements a full ML pipeline with:

-   **FastAPI Backend**: Serves predictions via a REST API
-   **Streamlit Frontend**: Provides a user-friendly web interface
-   **MLflow**: Tracks model training metrics and versions
-   **Docker**: Containerizes the entire application stack
-   **Scikit-learn**: Powers the ML model (Multinomial Naive Bayes)

Getting Started
---------------

### Prerequisites

-   Docker and Docker Compose
-   Python 3.11+
-   Git

### Local Development

1.  **Clone the repository**

```
git clone https://github.com/yourusername/language-detection.git
cd language-detection

```

1.  **Build and run with Docker Compose**

```
docker-compose up --build

```

This will start:

-   FastAPI backend on http://localhost:80
-   Streamlit frontend on http://localhost:8501
-   MLflow tracking server on http://localhost:5001

### Training the Model

The model is trained on a language detection dataset from Kaggle. To retrain the model:

```
python app/model/train_with_mlflow.py

```

This will:

-   Download the dataset using kagglehub
-   Train a MultinomialNB model with CountVectorizer
-   Track metrics with MLflow
-   Save the model pipeline for inference

API Endpoints
-------------

-   `GET /`: Health check and model version information
-   `POST /predict`: Predicts the language of input text
    -   Request body: `{"text": "Your text here"}`
    -   Response: `{"language": "English"}`

Deployment
----------

### Deploy to Heroku

```
heroku login
heroku create your-language-detector-app
heroku git:remote -a your-language-detector-app
heroku stack:set container
git push heroku main

```

Project Structure
-----------------

```
language-detection/
├── app/
│   ├── frontend/
│   │   └── streamlit_app.py    # Streamlit UI
│   ├── model/
│   │   ├── model.py            # Prediction code
│   │   └── train_with_mlflow.py # Training script
│   └── main.py                 # FastAPI application
├── notebooks/
│   └── language_detection.ipynb # Research notebook
├── Dockerfile                  # Container definition
├── docker-compose.yml         # Multi-container setup
└── requirements.txt           # Python dependencies

```

Future Improvements
-------------------

-   Add support for more languages
-   Improve model accuracy with more sophisticated features
-   Implement confidence scores for predictions
-   Add CI/CD pipeline for automated testing and deployment

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.
