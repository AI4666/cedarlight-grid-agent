FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
RUN useradd --create-home --uid 10001 agent
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-serve.txt
COPY . .
# Model artifacts are generated at build time (synthetic data, ~30 s) so the image never ships unverified binaries.
RUN cd 05_ml && python -I run_ml.py && chown -R agent:agent /app
USER agent
EXPOSE 8000
# Secrets (CEDARLIGHT_API_KEY, CEDARLIGHT_ADMIN_TOKEN) are supplied at runtime via env, never baked in.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)" || exit 1
CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
