FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/
COPY README.md .
COPY LICENSE .

RUN pip install --no-cache-dir -e ".[dev]"

# Run tests to verify the build
RUN pytest tests/ -v --tb=short

EXPOSE 8501

CMD ["streamlit", "run", "src/viz/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
