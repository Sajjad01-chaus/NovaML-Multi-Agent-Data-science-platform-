# One image for api, worker and ui; the command picks the role.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH
# Dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
RUN mkdir -p src/novaml && touch src/novaml/__init__.py \
 && pip install ".[server,groq,ui]" \
 && pip uninstall -y novaml
COPY src ./src
RUN pip install --no-deps .

FROM python:3.12-slim
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    NOVAML_DATA_DIR=/data
RUN useradd --create-home --uid 10001 novaml && mkdir /data && chown novaml /data
COPY --from=build /venv /venv
COPY ui /app/ui
WORKDIR /app
USER novaml
EXPOSE 8080 8501
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1
CMD ["novaml", "api", "--host", "0.0.0.0", "--port", "8080"]
