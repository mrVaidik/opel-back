import os
import logging
import subprocess
import json
import re
import uuid
import wave
import aiofiles
import aiohttp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from socketio import AsyncServer, ASGIApp
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel

# Load environment
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temp directory
TEMP_DIR = "temp_upload"
os.makedirs(TEMP_DIR, exist_ok=True)

# FastAPI app and Socket.IO
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = ASGIApp(socketio_server=sio, other_asgi_app=app)

# Initialize models
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
whisper_model = WhisperModel("small", device="cpu")

recorded_chunks = {}

@app.get("/test")
async def test():
    return {"status": "ok", "message": "Server is running"}

@sio.on("connect")
async def on_connect(sid, environ):
    logger.info(f"Connected: {sid}")
    recorded_chunks[sid] = []

@sio.on("disconnect")
async def on_disconnect(sid):
    logger.info(f"Disconnected: {sid}")
    recorded_chunks.pop(sid, None)

@sio.on("video-chunks")
async def receive_chunks(sid, data):
    filename = data["filename"]
    chunks = data["chunks"]
    if sid not in recorded_chunks:
        recorded_chunks[sid] = []
    recorded_chunks[sid].append(chunks)

    temp_path = os.path.join(TEMP_DIR, filename)
    async with aiofiles.open(temp_path, 'wb') as f:
        await f.write(b''.join(recorded_chunks[sid]))
    logger.info(f"Saved chunk: {filename}")

@sio.on("process-video")
async def process_video(sid, data):
    filename = data["filename"]
    user_id = data["userId"]
    video_path = os.path.join(TEMP_DIR, filename)
    audio_filename = f"audio_{uuid.uuid4().hex}.wav"
    audio_path = os.path.join(TEMP_DIR, audio_filename)
    max_size = int(os.getenv("MAX_FILE_SIZE", 25_000_000))

    try:
        file_size = os.path.getsize(video_path)
        if file_size > max_size:
            logger.warning(f"File too large: {file_size}")
            return

        ffmpeg_path = os.path.abspath('ffmpeg/bin/ffmpeg.exe')  # adjust if needed
        command = [ffmpeg_path, '-i', video_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Audio extracted: {audio_path}")

        segments, _ = whisper_model.transcribe(audio_path)
        transcription = " ".join([seg.text.strip() for seg in segments]).strip()

        logger.info(f"Transcription: {transcription}")
        if transcription:
            prompt = f"""
            You are going to generate a title and a nice description using the speech-to-text transcription provided.
            Transcription:
            {transcription}
            Return this JSON:
            {{
                "title": "<title>",
                "summary": "<summary>"
            }}
            """

            genai_model = genai.GenerativeModel("gemini-2.0-flash")
            result = genai_model.generate_content(prompt)
            content = result.text.strip()

            clean_json = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.DOTALL)
            parsed = json.loads(clean_json)
            logger.info(f"Gemini output: {parsed}")

            # You can store `parsed` somewhere or return it
            await sio.emit("video-processed", {
                "filename": filename,
                "transcription": transcription,
                "title": parsed["title"],
                "summary": parsed["summary"]
            }, room=sid)

    except Exception as e:
        logger.error(f"Processing error: {e}")
        await sio.emit("processing-error", {"message": str(e)}, room=sid)

    finally:
        for path in [video_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted: {path}")
        recorded_chunks.pop(sid, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=5000)
