FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi uvicorn requests

COPY ai_proxy.py /app/ai_proxy.py

EXPOSE 9000

CMD ["uvicorn", "ai_proxy:app", "--host", "0.0.0.0", "--port", "9000"]
