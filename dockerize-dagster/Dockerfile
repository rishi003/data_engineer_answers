FROM python:3.7-slim

RUN mkdir -p /opt/dagster/dagster_home /opt/dagster/app

RUN pip install dagit dagster pandas pyarrow joblib scikit-learn flask kaggle

# Copy your code and workspace to /opt/dagster/app
COPY workspace.yaml /opt/dagster/app/
COPY data_engineer_answers /opt/dagster/app/data_engineer_answers

ENV DAGSTER_HOME=/opt/dagster/dagster_home/

# Copy dagster instance YAML to $DAGSTER_HOME
COPY dagster.yaml /opt/dagster/dagster_home/

WORKDIR /opt/dagster/app

EXPOSE 3000

ENTRYPOINT ["dagit", "-h", "0.0.0.0", "-p", "3000"]