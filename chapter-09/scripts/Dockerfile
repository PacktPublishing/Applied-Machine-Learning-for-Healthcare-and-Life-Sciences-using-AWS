FROM python:3.7-slim-buster

RUN pip install pandas
RUN pip install wget
RUN pip install boto3
RUN pip install sagemaker
RUN pip install scikit-learn

# Make sure python doesn't buffer stdout so we get logs ASAP.
ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]