import os
import logging
import subprocess
import re
import json
import wave
import sys
import aiofiles
import aiohttp
import boto3
from botocore.config import Config
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from socketio import AsyncServer, ASGIApp
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import google.generativeai as genai
import ffmpeg
import uuid  # For generating unique filenames
from io import BytesIO
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (as before)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = ASGIApp(socketio_server=sio, other_asgi_app=app)


genai.configure(api_key="AIzaSyDtwjdZFSAW1pJSnAj1kURTN8kGeFWmATU")

# Initialize S3 client (as before)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("ACCESS_KEY"),
    aws_secret_access_key=os.getenv("SECRET_KEY"),
    region_name=os.getenv("BUCKET_REGION"),
    config=Config(signature_version='s3v4')
)

# Store recorded chunks per session
recorded_chunks = {}

# Load Vosk model on startup (optional, but can be more efficient)
vosk_model = None
try:
    vosk_model_path = 'vosk-model-en-us-0.22-lgraph'
    vosk_model = Model(vosk_model_path)
    logger.info("Vosk model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Vosk model: {e}")

# Test endpoint (as before)
@app.get("/test")
async def test_endpoint():
    return {"status": "ok", "message": "Server is running"}

# Socket.IO events
@sio.on("connect")
async def connect(sid, environ):
    logger.info(f"New client connected: {sid}, IP: {environ.get('REMOTE_ADDR')}, Headers: {environ}")
    recorded_chunks[sid] = []

@sio.on("disconnect")
async def disconnect(sid):
    logger.info(f"{sid} disconnected")
    if sid in recorded_chunks:
        del recorded_chunks[sid]

@sio.on("video-chunks")
async def handle_video_chunks(sid, data):
    try:
        filename = data['filename']
        chunks = data['chunks']
        if sid not in recorded_chunks:
            recorded_chunks[sid] = []
        recorded_chunks[sid].append(chunks)
        logger.info(f"Chunk received for session {sid}, filename: {filename}")
    except Exception as e:
        logger.error(f"Error handling video chunks for session {sid}: {e}")
        await sio.emit("upload_error", {"message": "Error receiving video chunk"}, room=sid)

async def extract_audio_memory(video_data):
    """Extracts audio from in-memory video data."""
    try:
        ffmpeg_path = "ffmpeg"  
        if not os.path.isfile(ffmpeg_path):
            logger.error(f"FFmpeg executable not found at: {ffmpeg_path}")
            return None

        process = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            '-i', '-',  # Input from stdin
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            '-',  # Output to stdout
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=video_data)

        if process.returncode == 0:
            logger.info(f"Audio extraction complete from in-memory video.")
            return stdout
        else:
            logger.error(f"Error during audio extraction from in-memory video: {stderr.decode('utf-8')}")
            return None
    except Exception as e:
        logger.error(f"Error during in-memory audio extraction: {e}")
        return None

async def transcribe_audio_memory(audio_data):
    """Transcribes audio data from memory."""
    if not vosk_model:
        logger.warning("Vosk model not loaded, skipping transcription.")
        return None

    try:
        with wave.open(BytesIO(audio_data), "rb") as wf:
            recognizer = KaldiRecognizer(vosk_model, wf.getframerate())
            result_text = ''
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    result_json = json.loads(result)
                    result_text += result_json.get('text', '') + ' '
            final_result = recognizer.FinalResult()
            final_result_json = json.loads(final_result)
            result_text += final_result_json.get('text', '') + ' '
            transcription = result_text.strip()
            logger.info(f"Transcription from in-memory audio: {transcription}")
            return transcription
    except wave.Error as e:
        logger.error(f"Error processing in-memory audio data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during in-memory transcription: {e}")
        return None

async def generate_title_summary(transcription):
    """Generates title and summary using Gemini."""
    if not transcription:
        return None

    gemini_prompt = f"""
        You are going to generate a title and a nice description using the speech-to-text transcription provided.

        Transcription:
        {transcription}

        Return the response in this JSON format:
        {{
            "title": "<generated title>",
            "summary": "<generated summary>"
        }}
        """
    try:
        genai_model = genai.GenerativeModel('gemini-2.0-flash')
        gemini_response = genai_model.generate_content(gemini_prompt)
        gemini_text = gemini_response.text.strip()
        try:
            gemini_text_clean = re.sub(r"^```json\s*|\s*```$", "", gemini_text.strip(), flags=re.DOTALL).strip()
            gemini_json = json.loads(gemini_text_clean)
            return gemini_json
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding Gemini JSON response: {e}, Response text: '{gemini_text}'")
            return None
    except Exception as e:
        logger.error(f"Error generating content with Gemini: {e}")
        return None

@sio.on("process-video")
async def process_video(sid, data):
    filename = data['filename']
    user_id = data['userId']
    max_file_size = int(os.getenv("MAX_FILE_SIZE", 25000000)) # Get limit from env

    try:
        if sid not in recorded_chunks or not recorded_chunks[sid]:
            logger.warning(f"No video chunks received for session {sid}, filename: {filename}. Skipping processing.")
            return

        video_data = b''.join(recorded_chunks[sid])
        file_size = len(video_data)

        if file_size > max_file_size:
            logger.warning(f"File size exceeds the limit ({max_file_size} bytes) for {filename}, skipping processing.")
            await sio.emit("upload_error", {"message": "File size too large"}, room=sid)
            return

        # Notify processing start
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('NEXT_API_HOST')}recording/{user_id}/processing",
                json={"filename": filename}
            ) as resp:
                processing_resp = await resp.json()
                if processing_resp.get('status') != 200:
                    logger.error(f"Error notifying processing start for user {user_id}: {processing_resp}")
                    return

        # Upload to S3 (upload the in-memory data directly)
        response = s3.put_object(
            Bucket=os.getenv("BUCKET_NAME"),
            Key=filename,
            ContentType='video/webm',
            Body=video_data
        )
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            logger.info(f"Video uploaded to AWS for user {user_id}, filename: {filename}")
        else:
            logger.error(f"Error uploading video to S3 for user {user_id}, filename: {filename}, status: {response['ResponseMetadata']['HTTPStatusCode']}")
            return

        if processing_resp.get('plan') == "PRO":
            audio_data = await extract_audio_memory(video_data)
            if audio_data:
                transcription = await transcribe_audio_memory(audio_data)
                if transcription:
                    gemini_output = await generate_title_summary(transcription)
                    if gemini_output:
                        async with aiohttp.ClientSession() as session:
                            transcribe_resp = await session.post(
                                f"{os.getenv('NEXT_API_HOST')}recording/{user_id}/transcribe",
                                json={
                                    "filename": filename,
                                    "content": gemini_output,
                                    "transcript": transcription
                                }
                            )
                            print(f"api send : {transcribe_resp}")
                            print("file successfully transfer")
                            if transcribe_resp.status == 200:
                                logger.info(f"Transcription data sent to Next API for {filename}")
                            else:
                                logger.error(f"Error sending transcription data to Next API for {filename}: {transcribe_resp.status} - {await transcribe_resp.text()}")
                    else:
                        logger.info(f"No title/summary generated for {filename}.")
                else:
                    logger.info(f"No transcription found for {filename}.")
            else:
                logger.error(f"Failed to extract audio from in-memory video for {filename}.")
        else:
            logger.info(f"User {user_id} is not on the 'PRO' plan, skipping transcription for {filename}.")

    except Exception as e:
        logger.error(f"Error processing video for {filename}: {e}")
        await sio.emit("processing_error", {"message": "Error processing video"}, room=sid)
    finally:
        # Notify completion for all users (PRO and non-PRO)
        try:
            async with aiohttp.ClientSession() as session:
                complete_resp = await session.post(
                    f"{os.getenv('NEXT_API_HOST')}recording/{user_id}/complete",
                    json={"filename": filename}
                )
                if complete_resp.status == 200:
                    logger.info(f"Successfully notified completion for {filename}")
                else:
                    logger.error(f"Error notifying completion for {filename}: {complete_resp.status} - {await complete_resp.text()}")
        except Exception as e:
            logger.error(f"Exception during completion callback for {filename}: {e}")

        # Clear the recorded chunks for the session
        if sid in recorded_chunks:
            del recorded_chunks[sid]

if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=5000)