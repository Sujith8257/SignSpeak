# SignSpeak - Indian Sign Language Recognition Dockerfile
# Uses Python 3.11 slim so pip-freeze requirements (e.g. contourpy>=1.3.3) install correctly

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies for OpenCV, MediaPipe, and camera access
# Note: On newer Debian (trixie), use libgl1 instead of deprecated libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p model checkpoints

# Create a non-root user and add to video group for camera access (Linux)
# This fixes camera permission issues when using --device=/dev/video0
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser || true && \
    usermod -aG video appuser || true && \
    chown -R appuser:appgroup /app

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/status', timeout=5)" || exit 1

# Run as non-root user (comment out if camera doesn't work)
# USER appuser

# Run the application
CMD ["python", "app.py"]
