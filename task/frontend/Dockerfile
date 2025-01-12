FROM python:3.11-slim

RUN apt-get -y update && \
    apt-get -y install python3-pip

# pip install the other python requirements in the container
WORKDIR .
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# copy the src code into the directory in the container
COPY . /web/

WORKDIR /web

ENV PYTHONUNBUFFERED 1

ENTRYPOINT gunicorn --worker-class uvicorn.workers.UvicornWorker \
  frontend_server:app \
  --workers 3 \
  --bind 0.0.0.0:8090 \
  --log-level info