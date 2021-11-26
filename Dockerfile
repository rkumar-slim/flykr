FROM python:3.8.6-buster
COPY api /api
COPY flykr /flykr
COPY requirements.txt /requirements.txt
COPY asl_model.h5 /asl_model.h5
COPY asl_labelbinarizer.h5 /asl_labelbinarizer.h5
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.api:app --host 0.0.0.0
