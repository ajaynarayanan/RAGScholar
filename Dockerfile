FROM python:3.12.6
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
CMD ["python", "src/server.py"]