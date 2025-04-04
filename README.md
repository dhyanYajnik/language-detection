# Language Detection API

A machine learning application that detects the language of input text. The system supports 17 languages including Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portuguese, Russian, Spanish, Swedish, Tamil, and Turkish.

## Project Architecture

This project implements a full ML pipeline with:

- **FastAPI Backend**: Serves predictions via a REST API
- **Streamlit Frontend**: Provides a user-friendly web interface
- **MLflow**: Tracks model training metrics and versions
- **Docker**: Containerizes the entire application stack
- **Scikit-learn**: Powers the ML model (Multinomial Naive Bayes)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git

### Local Development

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/language-detection.git
cd language-detection
