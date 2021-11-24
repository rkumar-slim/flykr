FROM python:3.8.6-buster
COPY api /api
COPY BaselineModel /BaselineModel
COPY requirements.txt /requirements.txt
COPY model.h5 /model.h5
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0
