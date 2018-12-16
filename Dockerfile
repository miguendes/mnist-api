FROM python:3.6.7

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install pipenv==2018.10.13

RUN pipenv install --system --deploy --skip-lock

ENTRYPOINT ["python", "app.py"]
