FROM python:3.10-alpine
RUN mkdir /App
WORKDIR /App
COPY . /App
RUN pip install -r requirements.txt
CMD ["sh", "-c", "python manage.py makemigrations && python manage.py migrate && python manage.py runserver"]
