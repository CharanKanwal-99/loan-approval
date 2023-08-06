FROM python:3.10-alpine
RUN mkdir /App
WORKDIR /App
COPY . /App
RUN pip install -r requirements.txt
CMD ["python","manage.py","runserver"]
