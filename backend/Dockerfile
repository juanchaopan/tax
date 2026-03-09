FROM python:3.13-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --only main --no-root --no-interaction

COPY backend.py ./
COPY chroma_db ./chroma_db

EXPOSE 8000

CMD ["python", "backend.py"]
