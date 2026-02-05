# Docker Deployment Guide

This guide explains how to deploy SignSpeak using Docker for easy deployment on any system.

## ⚠️ CRITICAL: Windows Camera Limitation

**If you're on Windows and need camera access for sign language detection:**

🚫 **Docker on Windows DOES NOT reliably work with webcams**

✅ **Use Python instead** (see [RUN-ON-ANOTHER-WINDOWS.md](RUN-ON-ANOTHER-WINDOWS.md))

**Why?** Docker Desktop runs Linux containers that cannot access Windows camera APIs. You will see "Camera Not Available" errors.

**Docker works fine with cameras on:**
- ✅ Linux
- ✅ macOS (usually)
- ❌ Windows (rarely works)

If you must use Docker on Windows, see [Windows Camera Access](#windows) section below.

---

## Prerequisites

- **Docker Desktop** installed and running ([Install Docker Desktop](https://docs.docker.com/get-docker/))
  - **Windows:** Make sure Docker Desktop is started (check system tray icon)
  - **Linux:** Docker daemon must be running (`sudo systemctl start docker`)
  - **macOS:** Docker Desktop must be running
- **Docker Compose** (usually included with Docker Desktop)
- **Camera access** (webcam or USB camera)

### Important: Start Docker Desktop First!

**Before running `docker-compose up`, ensure Docker Desktop is running:**

- **Windows:** Look for Docker Desktop icon in system tray (bottom-right). If not running, start it from Start menu.
- **Linux:** Run `sudo systemctl status docker` to check if Docker daemon is running.
- **macOS:** Check Docker Desktop menu bar icon. If not running, start it from Applications.

**Error you might see if Docker isn't running:**
```
unable to get image '...': error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/...": 
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

**Solution:** Start Docker Desktop and wait for it to fully initialize (whale icon should be steady, not animated).

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Place your model files** in the `model/` directory:
   ```
   model/
   ├── indian_sign_model.h5  (required)
   └── model_metadata.json   (optional)
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Open browser: http://localhost:5000
   - Allow camera access when prompted

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t signspeak:latest .
   ```

2. **Run the container:**
   
   **Linux:**
   ```bash
   docker run -d \
     --name signspeak-app \
     -p 5000:5000 \
     --device=/dev/video0 \
     -v $(pwd)/model:/app/model:ro \
     signspeak:latest
   ```
   
   **Windows (PowerShell):**
   ```powershell
   docker run -d `
     --name signspeak-app `
     -p 5000:5000 `
     -v ${PWD}/model:/app/model:ro `
     signspeak:latest
   ```
   
   **macOS:**
   ```bash
   docker run -d \
     --name signspeak-app \
     -p 5000:5000 \
     -v $(pwd)/model:/app/model:ro \
     signspeak:latest
   ```

3. **Access:** http://localhost:5000

4. **Stop:**
   ```bash
   docker stop signspeak-app
   docker rm signspeak-app
   ```

## Camera Access

### Linux (Camera Works!)

Use **`docker-compose.linux.yml`** for camera access on Linux:

```bash
# Check your camera device
ls -la /dev/video*

# Run with camera access
docker-compose -f docker-compose.linux.yml up --build

# Or use the script
chmod +x run-with-docker-linux.sh
./run-with-docker-linux.sh
```

If your camera is `/dev/video1` instead of `/dev/video0`, edit `docker-compose.linux.yml`:
```yaml
devices:
  - /dev/video1:/dev/video1  # Change to your device
```

**Key settings for camera access (already in docker-compose.linux.yml):**
```yaml
devices:
  - /dev/video0:/dev/video0
group_add:
  - video
```

The `group_add: video` adds the container user to the video group, fixing permission issues.

### Windows

**Important:** Windows cameras are **not easily accessible** from Linux Docker containers. Docker Desktop on Windows doesn't automatically bridge Windows cameras to Linux containers.

**Solutions:**

1. **Option A: Use privileged mode** (already enabled in docker-compose.yml)
   ```bash
   docker-compose up
   ```
   If this doesn't work, try Option B or C.

2. **Option B: Use host network mode** (modify docker-compose.yml):
   ```yaml
   network_mode: host
   ```
   Then remove the `ports` section (not needed with host network).

3. **Option C: Run natively on Windows** (recommended for Windows users)
   ```powershell
   # Don't use Docker - run directly
   python app.py
   ```
   This avoids camera access issues entirely.

4. **Option D: Use WSL2 with GUI** (advanced)
   - Install WSL2 with GUI support
   - Run Docker inside WSL2
   - Access Windows camera from WSL2

### macOS

macOS Docker Desktop typically handles camera access automatically. If issues occur:

1. Grant Docker Desktop camera permissions in System Preferences → Security & Privacy
2. Or use `--privileged` flag (less secure)

## Model Files

### Mounting Models (Recommended)

Models are mounted as read-only volumes, so you can update them without rebuilding:

```yaml
volumes:
  - ./model:/app/model:ro
```

Place your `.h5` model file in `./model/` on your host machine.

### Including Models in Image

If you prefer to include models in the Docker image:

1. Copy models to `model/` directory before building
2. Remove the `volumes` section from `docker-compose.yml`
3. Rebuild: `docker-compose build`

## Environment Variables

You can customize behavior via environment variables in `docker-compose.yml`:

```yaml
environment:
  - FLASK_ENV=production
  - PYTHONUNBUFFERED=1
  - TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow warnings
```

## Troubleshooting

### Camera Not Detected

**Linux:**
- Check camera device: `ls -la /dev/video*`
- Uncomment `devices` section in `docker-compose.yml` and set correct device path
- Try running with `--privileged` flag

**Windows (Common Issue):**
- **Error:** `VIDEOIO(V4L2:/dev/video0): can't open camera by index`
- **Cause:** Windows cameras use DirectShow/MSMF APIs, not accessible from Linux containers
- **Solutions:**
  1. **Best:** Run natively on Windows (not Docker):
     ```powershell
     python app.py
     ```
  2. **Alternative:** Use WSL2 with GUI support (advanced setup)
  3. **Workaround:** Try host network mode in docker-compose.yml:
     ```yaml
     network_mode: host
     ```
     (Remove `ports` section when using host network)
  4. **Last resort:** Use `--privileged` flag (already enabled in docker-compose.yml)

**macOS:**
- Ensure Docker Desktop has camera permissions (System Preferences → Security & Privacy)
- Try `--privileged` flag
- Check if camera works outside Docker first

### Model Not Found

- Ensure `model/indian_sign_model.h5` exists in `./model/` directory
- Check volume mount: `docker exec signspeak-app ls -la /app/model`
- Verify file permissions

### Port Already in Use

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "5001:5000"  # Use port 5001 instead
```

### Out of Memory

Adjust resource limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 4G  # Increase if needed
```

### View Logs

```bash
# Docker Compose
docker-compose logs -f

# Docker directly
docker logs -f signspeak-app
```

## Production Deployment

For production, consider:

1. **Use a reverse proxy** (nginx/traefik) in front of Flask
2. **Enable HTTPS** with SSL certificates
3. **Set resource limits** appropriately
4. **Use Docker secrets** for sensitive data
5. **Monitor with health checks** (already included)

## Building for Different Platforms

### Build for specific platform:

```bash
# Linux AMD64
docker build --platform linux/amd64 -t signspeak:latest .

# Linux ARM64 (for Raspberry Pi, etc.)
docker build --platform linux/arm64 -t signspeak:latest .
```

## Updating the Application

1. **Pull latest code** (if using git)
2. **Rebuild image:**
   ```bash
   docker-compose build --no-cache
   ```
3. **Restart:**
   ```bash
   docker-compose up -d
   ```

## Clean Up

Remove containers, images, and volumes:

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi signspeak:latest

# Clean up unused resources
docker system prune -a
```

## Additional Notes

- The Dockerfile uses Python 3.10 slim for smaller image size
- System dependencies are installed for OpenCV and MediaPipe
- Health check is configured to monitor container status
- Models are mounted as read-only for security
- Container runs as root (for camera access) - consider using `--user` flag for production if camera access allows
