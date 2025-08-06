# ---------- 🛠️ Stage 1: Pre-wheel Heavy Packages ----------
FROM python:3.11-bookworm AS wheel-prep
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libhdf5-dev \
    libsndfile1 \
    libffi-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Select heavyweight packages likely to slow builds
RUN mkdir /prewheels && pip wheel \
    torch torchvision faiss-cpu grpcio cryptography psycopg2-binary protobuf \
    --wheel-dir=/prewheels --no-cache-dir --verbose

# ---------- 🧱 Stage 2: Full Wheel Builder ----------
FROM python:3.11-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt .

# 📍 Diagnostic: Confirm lockfile visibility
RUN echo "🔍 Checking requirements.lock.txt..." && \
    ls -lh requirements.lock.txt && \
    cat requirements.lock.txt || echo "❌ Lockfile missing or unreadable"

# Copy prebuilt heavy wheels from previous stage
COPY --from=wheel-prep /prewheels /wheels

# Build remaining wheels not covered earlier
RUN pip wheel --wheel-dir=/wheels -r requirements.lock.txt --no-cache-dir --verbose

# 📍 Diagnostics for contributor clarity
RUN echo "📦 Wheel contents:" && ls -lh /wheels && \
    echo "📄 Build index:" && ls -1 /wheels > /wheels/build_index.txt && \
    cat /wheels/build_index.txt

# ---------- 🚀 Final Runtime ----------
FROM python:3.11-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    libhdf5-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use fully built wheels for pip install
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && \
    pip list --format=columns > /wheels/pip_installed.txt

# App source
COPY backend/app ./app
COPY backend/start.py ./start.py
COPY requirements.lock.txt .

# 🩺 Startup diagnostics
RUN echo "🏁 Checking backend state..." && \
    test -f start.py && echo "✅ start.py ready" || echo "❌ Missing start.py"

CMD ["python", "start.py"]
