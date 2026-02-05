#!/bin/bash
# SignSpeak - Run with Docker on Linux (with camera access)
# Usage: chmod +x run-with-docker-linux.sh && ./run-with-docker-linux.sh

echo ""
echo "============================================"
echo "  SignSpeak - Starting with Docker (Linux)"
echo "============================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker is not running."
    echo "Start Docker daemon: sudo systemctl start docker"
    exit 1
fi

# Check camera device
if [ ! -e /dev/video0 ]; then
    echo "[WARNING] /dev/video0 not found."
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "  (none found)"
    echo ""
    echo "If your camera is /dev/video1, edit docker-compose.linux.yml"
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

# Check model exists
if [ ! -f "model/indian_sign_model.h5" ]; then
    echo "[WARNING] model/indian_sign_model.h5 not found."
    echo "Copy your .h5 model file into the model/ folder."
    echo ""
fi

echo "Starting Docker containers with camera access..."
echo ""
echo "When the server is ready, open:  http://localhost:5000"
echo "Press Ctrl+C to stop."
echo ""

# Run with Linux camera config
docker-compose -f docker-compose.linux.yml up --build
