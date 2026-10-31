# Forecast API image: serves the forecasts made by src.forecast.
#
#   python -m src.forecast                       # make output/forecast/ first
#   docker build -t hayfever-api .
#   docker run -p 8000:8000 hayfever-api         # then open http://127.0.0.1:8000/docs
#
# Only the API's packages are installed (no LightGBM or PyTorch): training happens outside
# the container, and a new forecast means a new image.
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
