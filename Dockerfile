# Serves output/forecast/ (run python -m src.forecast first)
#   docker build -t hayfever-api .
#   docker run -p 8000:8000 hayfever-api
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -c requirements.txt pandas numpy fastapi uvicorn

COPY src/__init__.py src/api.py src/
COPY output/forecast/ output/forecast/

RUN useradd --create-home api
USER api
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
