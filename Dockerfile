FROM python:3.11.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]

# docker build -t language-detection-app .  
# docker run -p 80:80 language-detection-app

# docker compose down
# docker compose up --build