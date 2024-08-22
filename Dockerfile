# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de la aplicación al contenedor
COPY app.py /app/

# Crear el directorio models y copiar el modelo
RUN mkdir -p /app/data
COPY data/heart.csv /app/data/

# Crear el directorio models y copiar el modelo
RUN mkdir -p /app/models
COPY models/pipeline.pkl /app/models/
COPY models/training.ipynb /app/models/

# COPY models/preprocessor.pkl /app/models/

# Crear el directorio models y copiar el modelo
RUN mkdir -p /app/tests
COPY tests/test_app.py /app/tests/
COPY tests/test_pipeline.py /app/tests/



# Instalar las dependencias necesarias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8080

# Añadir un healthcheck al Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl --fail http://localhost:8080/ || exit 1

# Comando por defecto para ejecutar la aplicación
CMD ["python", "app.py"]

