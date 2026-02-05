# Real-Time Lipsync Streaming Server (Python + WebRTC)

This project is a high-performance streaming architecture designed to serve ML-generated video frames (e.g., lipsync models) to a browser with ultra-low latency and perfect Audio/Video synchronization.

It solves the "Chipmunk Effect" (speed-up) and Audio Drift issues common in ML streaming by decoupling **Generation Speed** from **Playback Speed** using a custom 3-Stage Pipeline.

## 🏗 Architecture

The system uses **WebRTC (`aiortc`)** for the transport layer, but introduces a custom "Valve" mechanism to handle variable inference speeds.

### The 3-Stage Pipeline

1.  **Stage 1: The Producer (Fast)**
    - **Role:** Simulates the ML Model. Generates frames as fast as possible (e.g., 50+ FPS).
    - **Audio Logic:** Uses a "Meat Slicer" (FIFO Buffer) to slice audio into exact 40ms chunks (1920 samples) to match the Video FPS, preventing drift over time.
    - **Output:** Dumps data into `Buffer A` (Unlimited size).

2.  **Stage 2: The Pacer (The Valve)**
    - **Role:** The heartbeat of the system. It moves data from `Buffer A` to `Buffer B`.
    - **Video Logic:** Strictly enforces 25 FPS (sleeps until the exact millisecond a frame is due).
    - **Audio Logic:** "Leads" the video by ~12 frames (0.5s). This fills the browser's audio buffer to prevent crackling while keeping lip movements perfectly synced.

3.  **Stage 3: The Consumer (Network)**
    - **Role:** `aiortc` tracks that read from `Buffer B`.
    - **Behavior:** If the Pacer hasn't released a frame yet, the network track blocks/waits. This physically forces the browser to play at 1x speed.

---

## 🚀 Installation

### 1. Prerequisites

- **Miniconda** (Recommended for macOS Apple Silicon to handle binary dependencies).
- **FFmpeg** (For generating test data).

### 2. Setup Environment

We use Conda to avoid compilation errors with `av` and `aiortc` on M1/M2/M3 Macs.

```bash
# 1. Create a clean environment
conda create -n lipsync python=3.10 -y
conda activate lipsync

# 2. Install heavy dependencies via Conda (Pre-compiled binaries)
conda install -c conda-forge av opencv aiohttp -y

# 3. Install aiortc via pip (Links correctly to Conda libs)
pip install aiortc

```

### 3. Generate Test Data

The server requires raw frames and audio to simulate the ML model input.

```bash
# Create the folder structure
mkdir -p test_data/frames

# Extract Audio (Must be 48k Hz)
ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 48000 -ac 2 test_data/audio.wav

# Extract Video Frames (Must match the FPS in server.py, default 25)
ffmpeg -i input_video.mp4 -vf fps=25 test_data/frames/frame_%04d.jpg

```

---

## 🏃‍♂️ Usage

1. **Start the Server:**

```bash
python server.py

```

2. **Open the Client:**

- Open your browser (Chrome/Safari) to: `http://localhost:8080`
- Click **Start Streaming**.

3. **Verify Pacing:**
   Check your terminal logs. You should see the Pacer heartbeat ticking exactly once per second, proving the stream is stable regardless of generation speed.

```text
PACER HEARTBEAT: Streaming 1s
PACER HEARTBEAT: Streaming 2s
PACER HEARTBEAT: Streaming 3s

```

---

## ⚙️ Configuration (`server.py`)

You can tune the streaming parameters at the top of the script:

```python
FPS = 25                  # Target playback FPS
AUDIO_RATE = 48000        # Audio Sample Rate
AUDIO_BUFFER_AHEAD = 12   # How many audio frames to pre-buffer (prevent crackling)

```
