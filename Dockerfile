# 1. Base Image: Usa l'immagine ufficiale PyTorch con CUDA 12.1 pre-installato
# Questo evita di scaricare gigabyte di wheel e risolve conflitti hardware
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Imposta variabili d'ambiente per evitare blocchi interattivi su Linux
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Installa le librerie C++ di sistema necessarie per OpenCV (Punto critico)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Imposta la directory di lavoro dentro il container
WORKDIR /app

# 5. Copia solo il requirements per sfruttare la cache di Docker
COPY requirements.txt .

# 6. Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copia tutto il resto del codice sorgente nel container
COPY . .

# 8. Definisci il comando di default per eseguire l'inferenza
# Il container agirà come se fosse un eseguibile del tuo predict.py
ENTRYPOINT ["python", "src/inference/predict.py"]