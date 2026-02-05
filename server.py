import asyncio
import logging
import os
import time
import fractions
import cv2
import av

from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    AudioStreamTrack,
)
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RTC")

# ================= CONFIG =================
FPS = 25
VIDEO_CLOCK = 90000
VIDEO_FRAME_TICKS = VIDEO_CLOCK // FPS

AUDIO_RATE = 48000
AUDIO_FRAME_MS = 20
AUDIO_SAMPLES = int(AUDIO_RATE * AUDIO_FRAME_MS / 1000)

FRAME_DIR = "test_data/frames"
AUDIO_PATH = "test_data/audio.wav"
# ==========================================


# ============ CONSUMER TRACKS ==============
class ConsumerVideoTrack(VideoStreamTrack):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    async def recv(self):
        frame = await self.queue.get()
        if frame is None:
            self.stop()
            raise MediaStreamError()
        return frame


class ConsumerAudioTrack(AudioStreamTrack):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    async def recv(self):
        frame = await self.queue.get()
        if frame is None:
            self.stop()
            raise MediaStreamError()
        return frame
# ===========================================


# ============ FAST PRODUCER =================
async def fast_producer(raw_audio_q, raw_video_q):
    logger.info("Producer: running FAST")

    frames = sorted(
        os.path.join(FRAME_DIR, f)
        for f in os.listdir(FRAME_DIR)
        if f.endswith(".jpg")
    )
    num_video_frames = len(frames)
    video_duration_sec = num_video_frames / FPS
    total_audio_samples = int(video_duration_sec * AUDIO_RATE)

    logger.info(
        f"Producer: {num_video_frames} video frames → "
        f"{video_duration_sec:.2f}s total"
    )

    container = av.open(AUDIO_PATH)
    audio_stream = container.streams.audio[0]

    resampler = av.AudioResampler(
        format="s16",
        layout="stereo",
        rate=AUDIO_RATE,
    )
    fifo = av.AudioFifo()

    video_idx = 0
    video_pts = 0
    audio_pts = 0

    for packet in container.demux(audio_stream):
        for raw in packet.decode():
            for frame in resampler.resample(raw):
                fifo.write(frame)

            while fifo.samples >= AUDIO_SAMPLES:
                if audio_pts >= total_audio_samples:
                    break

                audio = fifo.read(AUDIO_SAMPLES)
                audio.pts = audio_pts
                audio.time_base = fractions.Fraction(1, AUDIO_RATE)
                audio_pts += AUDIO_SAMPLES

                await raw_audio_q.put(audio)

                if video_idx < num_video_frames:
                    img = cv2.imread(frames[video_idx])
                    video_idx += 1

                    video = VideoFrame.from_ndarray(img, format="bgr24")
                    video.pts = video_pts
                    video.time_base = fractions.Fraction(1, VIDEO_CLOCK)
                    video_pts += VIDEO_FRAME_TICKS

                    await raw_video_q.put(video)

                # simulate fast inference
                await asyncio.sleep(0.001)

        if audio_pts >= total_audio_samples:
            break

    await raw_audio_q.put(None)
    await raw_video_q.put(None)

    logger.info(
        f"Producer finished: "
        f"{video_idx} video frames, "
        f"{audio_pts / AUDIO_RATE:.2f}s audio"
    )
# ===========================================


# ============ AUDIO-CLOCK PACER =============
async def audio_clock_pacer(raw_audio_q, raw_video_q, out_audio_q, out_video_q):
    logger.info("Pacer: audio clock master")

    start_time = time.time()
    audio_samples_sent = 0
    video_buffer = []

    audio_done = False
    video_done = False

    while True:
        # ---- AUDIO ----
        if not audio_done:
            audio = await raw_audio_q.get()
            if audio is None:
                audio_done = True
            else:
                target_time = start_time + (audio_samples_sent / AUDIO_RATE)
                await asyncio.sleep(max(0, target_time - time.time()))

                await out_audio_q.put(audio)
                audio_samples_sent += audio.samples

        # ---- VIDEO INGEST ----
        while not raw_video_q.empty():
            v = await raw_video_q.get()
            if v is None:
                video_done = True
            else:
                video_buffer.append(v)

        # ---- VIDEO RELEASE ----
        current_time = audio_samples_sent / AUDIO_RATE
        while video_buffer:
            next_v = video_buffer[0]
            v_time = float(next_v.pts * next_v.time_base)
            if v_time <= current_time:
                await out_video_q.put(video_buffer.pop(0))
            else:
                break

        # ---- EXIT ----
        if audio_done and video_done and not video_buffer:
            await out_audio_q.put(None)
            await out_video_q.put(None)
            break
# ===========================================


# ============ WEBRTC SERVER =================
pcs = set()

async def index(request):
    return web.Response(
        content_type="text/html",
        text=open("index.html").read(),
    )


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(**params)

    pc = RTCPeerConnection()
    pcs.add(pc)

    raw_audio_q = asyncio.Queue()
    raw_video_q = asyncio.Queue()
    out_audio_q = asyncio.Queue()
    out_video_q = asyncio.Queue()

    pc.addTrack(ConsumerAudioTrack(out_audio_q))
    pc.addTrack(ConsumerVideoTrack(out_video_q))

    asyncio.create_task(fast_producer(raw_audio_q, raw_video_q))
    asyncio.create_task(audio_clock_pacer(
        raw_audio_q, raw_video_q,
        out_audio_q, out_video_q
    ))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


if __name__ == "__main__":
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    print("Server running at http://localhost:8080")
    web.run_app(app, port=8080)