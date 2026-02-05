import argparse
import asyncio
import logging
import os
import cv2
import av
import fractions
import time

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from av import VideoFrame, AudioFrame
from aiortc.mediastreams import MediaStreamError # <--- IMPORT THIS

# Enable logging so you can SEE the pacing happen in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PACER")

# --- CONFIGURATION ---
FPS = 25
AUDIO_RATE = 48000
CHUNK_SIZE = int(AUDIO_RATE / FPS)  # 1920 samples (40ms)
FRAME_DIR = "test_data/frames"
AUDIO_PATH = "test_data/audio.wav"

# --- 1. THE CONSUMER TRACKS (Stage 3) ---
# These just read from Buffer B (Slow Queue).
# If Buffer B is empty, these NATURALLY wait.
class ConsumerVideoTrack(VideoStreamTrack):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    async def recv(self):
        # This blocks until the Pacer puts a frame in Buffer B
        frame = await self.queue.get() 
        if frame is None:
            self.stop()
            raise MediaStreamError() # <--- Graceful Stop
        return frame

class ConsumerAudioTrack(AudioStreamTrack):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    async def recv(self):
        frame = await self.queue.get()
        if frame is None:
            self.stop()
            raise MediaStreamError() # <--- Graceful Stop
        return frame

# --- 2. THE PACER (Stage 2: The Valve) ---
# This task moves data from A -> B at STRICT 25 FPS
async def strict_pacer(raw_v_q, raw_a_q, slow_v_q, slow_a_q):
    print(">>> Pacer Started: Clock initialized.")
    
    start_time = time.time()
    frame_count = 0
    frame_duration = 1.0 / FPS

    while True:
        # 1. Fetch from Buffer A (Instant)
        v_frame = await raw_v_q.get()
        a_frame = await raw_a_q.get()

        if v_frame is None: 
            # End of Stream logic
            await slow_v_q.put(None)
            await slow_a_q.put(None)
            print(">>> Pacer: Stream Complete.")
            break

        # 2. HEARTBEAT LOG (Verify Speed)
        # Every 25 frames (1 second), print status
        if frame_count % 25 == 0:
            print(f"PACER HEARTBEAT: Streaming {frame_count/25:.0f}s / {time.time()-start_time:.1f}s real-time")

        # 3. Wait for Target Time
        target_time = start_time + (frame_count * frame_duration)
        wait_time = target_time - time.time()
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # 4. Release to Buffer B
        await slow_v_q.put(v_frame)
        await slow_a_q.put(a_frame)
        
        frame_count += 1

# --- 3. THE PRODUCER (Stage 1: The Fast Model) ---
async def fast_producer(raw_v_q, raw_a_q):
    logger.info(">>> Producer: Generating as fast as possible...")
    
    container = av.open(AUDIO_PATH)
    audio_stream = container.streams.audio[0]
    
    # 1. PREPARE TOOLS
    # Resampler: Converts whatever the wav file is -> Standard WebRTC Format (s16, 48k, stereo)
    resampler = av.AudioResampler(format='s16', layout='stereo', rate=AUDIO_RATE)
    # FIFO: The "Meat Slicer" to get exact 40ms chunks
    fifo = av.AudioFifo()
    
    frame_files = sorted([os.path.join(FRAME_DIR, f) for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])
    total_frames = len(frame_files)
    frame_idx = 0
    pts_counter = 0

    for packet in container.demux(audio_stream):
        for raw_frame in packet.decode():
            
            # --- FIX: RESAMPLE BEFORE BUFFERING ---
            # We must convert the raw frame (which might be Planar) to Packed s16
            # before putting it in the FIFO.
            cleaned_frames = resampler.resample(raw_frame)
            
            for clean_frame in cleaned_frames:
                fifo.write(clean_frame)
            
            # --- SLICE IT (Read exact chunks) ---
            while fifo.samples >= CHUNK_SIZE:
                if frame_idx >= total_frames: break

                # Read 1920 samples (Exact 40ms)
                audio_frame = fifo.read(CHUNK_SIZE)
                
                # --- READ VIDEO FRAME ---
                img = cv2.imread(frame_files[frame_idx])
                frame_idx += 1

                # --- SET TIMESTAMPS ---
                audio_frame.pts = pts_counter * CHUNK_SIZE
                audio_frame.time_base = fractions.Fraction(1, AUDIO_RATE)
                
                video_frame = VideoFrame.from_ndarray(img, format="bgr24")
                video_frame.pts = pts_counter * 3600
                video_frame.time_base = fractions.Fraction(1, 90000)

                # --- PUSH TO BUFFER A ---
                await raw_a_q.put(audio_frame)
                await raw_v_q.put(video_frame)
                pts_counter += 1
                
                # Simulate Inference Speed
                await asyncio.sleep(0.001)
        
        if frame_idx >= total_frames: break

    # Signal End
    await raw_a_q.put(None)
    await raw_v_q.put(None)
    logger.info(f">>> Producer: Finished. Generated {frame_idx} frames.")
# --- 4. SERVER SETUP ---
async def index(request):
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Buffer A: Unlimited storage for the fast model
    raw_v_q = asyncio.Queue()
    raw_a_q = asyncio.Queue()

    # Buffer B: The output buffer that the Pacer feeds
    slow_v_q = asyncio.Queue()
    slow_a_q = asyncio.Queue()

    # Tracks read from Buffer B
    pc.addTrack(ConsumerVideoTrack(slow_v_q))
    pc.addTrack(ConsumerAudioTrack(slow_a_q))

    # Start the machinery
    # 1. Start Producer (Fills Buffer A)
    asyncio.create_task(fast_producer(raw_v_q, raw_a_q))
    # 2. Start Pacer (Moves A -> B slowly)
    asyncio.create_task(strict_pacer(raw_v_q, raw_a_q, slow_v_q, slow_a_q))

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

pcs = set()

if __name__ == "__main__":
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    print("Server running at http://localhost:8080")
    web.run_app(app, port=8080)