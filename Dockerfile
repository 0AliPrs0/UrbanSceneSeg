# --- Base Environment ---
FROM python:3.11-slim

# --- Working directory ---
WORKDIR /app

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Copy dependency list ---
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy all project files ---
COPY . .

# --- Default command (train model) ---
CMD ["python", "src/train.py"]
