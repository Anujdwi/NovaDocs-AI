# backend/Dockerfile — use paths relative to build context (./backend)
FROM python:3.11-slim

WORKDIR /app

# copy requirements.txt located at ./backend/requirements.txt on host
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the backend directory
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000"]
