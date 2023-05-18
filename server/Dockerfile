FROM python:3.9

WORKDIR /app

COPY server/main.py .
COPY data/model.joblib ./data/model.joblib

RUN pip install --no-cache-dir flask scikit-learn joblib

EXPOSE 5000

CMD ["python", "main.py"]
