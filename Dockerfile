FROM python:3.12-slim

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# install packages by conda
RUN pip install -r requirements.txt
CMD ["python", "Exercise_protected_server_TRIAL2.py"]
