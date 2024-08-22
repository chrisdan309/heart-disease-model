FROM python:3.9-slim

# Establecer workdir
WORKDIR /app

# Instalar curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de la aplicación al contenedor
COPY app.py /app/

# Copiar data
RUN mkdir -p /app/data
COPY data/heart.csv /app/data/

# Copiar modelos
RUN mkdir -p /app/models
COPY models/pipeline.pkl /app/models/
COPY models/training.ipynb /app/models/

# Copiar pruebas
RUN mkdir -p /app/tests
COPY tests/test_app.py /app/tests/
COPY tests/test_pipeline.py /app/tests/

# Instalar dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
# EXPOSE 8080

# Healthcheck (monitoreo)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl --fail http://localhost:${PORT:-8080}/ || exit 1

# Comando
CMD ["python", "app.py"]

