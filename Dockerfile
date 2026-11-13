FROM python:3.10-slim

WORKDIR /app

# Actualizar el gestor de paquetes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de dependencias primero para aprovechar el caché de Docker
COPY requirements.txt .

# Instalar PyTorch (versión CPU para agilizar la construcción, a menos que requieras GPU) y dependencias
RUN pip --default-timeout=1000 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del proyecto
COPY . .
