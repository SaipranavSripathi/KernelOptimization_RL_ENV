FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERNEL_ENV_MODE=surrogate

WORKDIR /app

COPY requirements-space.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-space.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "space_app:app", "--host", "0.0.0.0", "--port", "7860"]
