# Run SignSpeak on Another PC

Use this guide to run the project on a **different computer** with minimal setup.

> **On Linux?** Docker works great with camera! Use: `docker-compose -f docker-compose.linux.yml up --build`

---

## ⚠️ IMPORTANT: Docker vs Python on Windows

**For Windows users who need camera access (sign language detection):**

| Method | Camera Works? | When to Use |
|--------|---------------|-------------|
| **Option B: Python** | ✅ YES | **RECOMMENDED** - Use this if you need the webcam |
| **Option A: Docker** | ❌ NO (usually) | Only for testing without camera |

**Why Docker camera doesn't work on Windows:**
- Docker Desktop runs Linux containers on Windows
- Linux containers cannot access Windows camera APIs (DirectShow/MSMF)
- Result: You'll see "Camera Not Available" error
- **This is a limitation of Docker on Windows, not a bug in this project**

**Bottom line: If you need the camera to work, use Option B (Python).**

---

## What You Need to Copy

Copy the **entire project folder** to the other PC (USB drive, cloud, or network). It must include:

| Required | Purpose |
|----------|--------|
| `app.py` | Main application |
| `templates/` | Web interface (index.html) |
| `static/` | Static assets (e.g. dictionary.json) |
| `model/` | **Must contain `indian_sign_model.h5`** (put the .h5 file inside `model/`) |
| `requirements.txt` | Python dependencies (for Option B) |
| `Dockerfile` | For Docker build |
| `docker-compose.yml` | For Docker run |
| `.dockerignore` | For Docker build |

| Optional | Purpose |
|----------|--------|
| `checkpoints/` | If you use a checkpoint model |
| `RUN-ON-ANOTHER-WINDOWS.md` | This guide |
| `run-with-docker.bat` | One-click Docker start (Windows) |
| `run-with-python.bat` | One-click Python start (Windows, camera works) |
| `docker-compose.linux.yml` | For Linux with camera access |
| `run-with-docker-linux.sh` | One-click Docker start (Linux, camera works) |

**Important:** Create a `model` folder and put your trained model file inside it:
```
model/
  indian_sign_model.h5
```

---

## Option A: Run with Docker (no Python install)

Best for: quick run, no software install except Docker.  
**Limitation:** Webcam often does **not** work inside Docker on Windows (known Windows + Docker limitation).

### On the new Windows PC

1. **Install Docker Desktop**
   - Download: https://www.docker.com/products/docker-desktop/
   - Install and restart if asked.
   - Open Docker Desktop and wait until it says “Docker Desktop is running”.

2. **Open the project folder**
   - Go to the folder you copied (e.g. `sign-to-text-and-speech`).

3. **Start the app**
   - **Easy:** Double‑click `run-with-docker.bat`  
   - **Or:** Open PowerShell in that folder and run:
     ```powershell
     docker-compose up --build
     ```

4. **Open in browser**
   - Go to: **http://localhost:5000**

5. **Stop**
   - In the window that opened, press `Ctrl+C`, or close the window.

If the camera doesn’t work (e.g. “Camera Not Available”), use **Option B** on that PC for full camera support.

---

## Option B: Run with Python (camera works)

Best for: **using the webcam** on that Windows PC. Requires a one‑time Python install.

### On the new Windows PC

1. **Install Python 3.11**
   - Download: https://www.python.org/downloads/
   - Run the installer.
   - **Check “Add Python to PATH”**, then finish.

2. **Open the project folder**
   - Go to the folder you copied.

3. **Install dependencies (once per PC)**
   - Open PowerShell in that folder and run:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
   - If you get an execution policy error, run:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
     Then run the `Activate.ps1` and `pip install` again.

4. **Start the app**
   - **Easy:** Double‑click `run-with-python.bat`  
   - **Or:** In PowerShell (with venv activated):
     ```powershell
     python app.py
     ```

5. **Open in browser**
   - Go to: **http://localhost:5000**

6. **Stop**
   - In the terminal, press `Ctrl+C`.

---

## Option C: Run on Linux (Docker + Camera Works!)

If the other PC runs **Linux** (Ubuntu, Debian, etc.), Docker can access the webcam directly.

### On the Linux PC

1. **Install Docker** (if not installed):
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo usermod -aG docker $USER
   # Log out and back in for group change to take effect
   ```

2. **Copy the project folder** (including `model/indian_sign_model.h5`).

3. **Check your camera device:**
   ```bash
   ls -la /dev/video*
   ```

4. **Run with camera access:**
   ```bash
   cd sign-to-text-and-speech
   chmod +x run-with-docker-linux.sh
   ./run-with-docker-linux.sh
   ```
   Or manually:
   ```bash
   docker-compose -f docker-compose.linux.yml up --build
   ```

5. **Open in browser:** http://localhost:5000

6. **Stop:** Press `Ctrl+C`.

If your camera is `/dev/video1` instead of `/dev/video0`, edit `docker-compose.linux.yml` and change the device path.

---

## Quick Reference

| You Want To... | Use This |
|------|-----|
| ✅ **Run with webcam on Windows** | **Option B (Python) - RECOMMENDED** |
| ✅ **Run with webcam on Linux** | **Option C (Docker + docker-compose.linux.yml)** |
| ❌ Run without camera / test only | Option A (Docker on Windows) |
| Run on Mac with webcam | Option A (Docker) usually works on Mac |

**TL;DR:** On Windows, Docker camera = broken. Use Python. On Linux, Docker camera works!

---

## Troubleshooting

- **Docker: “Cannot connect to Docker daemon”**  
  Start Docker Desktop and wait until it’s fully running.

- **Docker: “Camera Not Available”**  
  Normal on Windows when using Docker. Use Option B (Python) on that PC for the camera.

- **Python: “python is not recognized”**  
  Install Python and make sure “Add Python to PATH” was checked. Restart PowerShell.

- **Python: “No module named …”**  
  Activate the venv and run `pip install -r requirements.txt` again.

- **“Model not found”**  
  Ensure the `model` folder exists and contains `indian_sign_model.h5` in the project folder.
