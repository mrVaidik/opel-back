import os
import logging
import subprocess
import re
import json
import wave
import sys
import asyncio
import aiofiles
import aiohttp
import boto3
from botocore.config import Config
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import google.generativeai as genai
import ffmpeg
import uuid  # For generating unique filenames

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Socket.IO
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins='*')

genai.configure(api_key="AIzaSyDtwjdZFSAW1pJSnAj1kURTN8kGeFWmATU")

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("ACCESS_KEY"),
    aws_secret_access_key=os.getenv("SECRET_KEY"),
    region_name=os.getenv("BUCKET_REGION"),
    config=Config(signature_version='s3v4')
)

# Create temp_upload directory if it doesn't exist
if not os.path.exists('temp_upload'):
    os.makedirs('temp_upload')

# Store recorded chunks per session
recorded_chunks = {}

# Load Vosk model on startup
vosk_model = None
try:
    vosk_model_path = 'vosk-model-en-us-0.22-lgraph'
    vosk_model = Model(vosk_model_path)
    logger.info("Vosk model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Vosk model: {e}")

# Test endpoint
@app.route("/test", methods=['GET'])
def test_endpoint():
    return jsonify({"status": "ok", "message": "Server is running"})

# Socket.IO events
@socketio.on("connect")
def connect():
    sid = request.sid
    environ = request.environ
    logger.info(f"New client connected: {sid}, IP: {environ.get('REMOTE_ADDR')}, Headers: {environ}")
    recorded_chunks[sid] = []

@socketio.on("disconnect")
def disconnect():
    sid = request.sid
    logger.info(f"{sid} disconnected")
    if sid in recorded_chunks:
        del recorded_chunks[sid]

@socketio.on("video-chunks")
def handle_video_chunks(data):
    sid = request.sid
    try:
        filename = data['filename']
        chunks = data['chunks']
        if sid not in recorded_chunks:
            recorded_chunks[sid] = []
        recorded_chunks[sid].append(chunks)

        temp_file_path = f'temp_upload/{filename}'
        with open(temp_file_path, 'wb') as f:
            f.write(b''.join(recorded_chunks[sid]))
        logger.info(f"Chunk received and saved for session {sid}, filename: {filename}")
    except Exception as e:
        logger.error(f"Error handling video chunks for session {sid}: {e}")
        emit("upload_error", {"message": "Error saving video chunk"}, room=sid)

@socketio.on("process-video")
def process_video(data):
    sid = request.sid
    filename = data['filename']
    user_id = data['userId']
    temp_video_path = f'temp_upload/{filename}'
    unique_audio_filename = f"audio_{uuid.uuid4().hex}.wav"
    temp_audio_path = f'temp_upload/{unique_audio_filename}'
    max_file_size = int(os.getenv("MAX_FILE_SIZE", 25000000)) # Get limit from env

    async def process():
        try:
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

            # Upload to S3
            async with aiofiles.open(temp_video_path, 'rb') as f:
                file_content = await f.read()
                response = s3.put_object(
                    Bucket=os.getenv("BUCKET_NAME"),
                    Key=filename,
                    ContentType='video/webm',
                    Body=file_content
                )
                if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                    logger.info(f"Video uploaded to AWS for user {user_id}, filename: {filename}")
                else:
                    logger.error(f"Error uploading video to S3 for user {user_id}, filename: {filename}, status: {response['ResponseMetadata']['HTTPStatusCode']}")
                    return

            if processing_resp.get('plan') == "PRO":
                file_size = os.path.getsize(temp_video_path)
                if file_size < max_file_size:
                    ffmpeg_path = os.path.abspath('ffmpeg-2025-03-31-git-35c091f4b7-essentials_build/bin/ffmpeg.exe')
                    if not os.path.isfile(ffmpeg_path):
                        logger.error(f"FFmpeg executable not found at: {ffmpeg_path}")
                        return

                    command = [ffmpeg_path, '-i', temp_video_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio_path]
                    try:
                        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        logger.info(f"Audio extraction complete for {filename}, audio saved to {temp_audio_path}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error during audio extraction for {filename}: {e.stderr.decode('utf-8')}")
                        return

                    if not os.path.exists(temp_audio_path):
                        logger.error(f"Audio file does not exist at {temp_audio_path} for {filename}")
                        return

                    try:
                        with wave.open(temp_audio_path, "rb") as wf:
                            if vosk_model:
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
                                logger.info(f"Transcription for {filename}: {transcription}")

                                if transcription:
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

                                        gemini_text = gemini_response.text.strip() # Get text and strip whitespace
                                        print(gemini_text)
                                        try:
                                            gemini_text_clean = re.sub(r"^```json\s*|\s*```$", "", gemini_text.strip(), flags=re.DOTALL).strip()
                                            gemini_json = json.loads(gemini_text_clean)
                                            print(gemini_json)
                                            print("before async")
                                            async with aiohttp.ClientSession() as session:
                                                transcribe_resp = await session.post(
                                                    f"{os.getenv('NEXT_API_HOST')}recording/{user_id}/transcribe",
                                                    json={
                                                        "filename": filename,
                                                        "content": gemini_json,
                                                        "transcript": transcription
                                                    }
                                                )
                                                print(f"api send : {transcribe_resp}")
                                                print("file successfully transfer")
                                                if transcribe_resp.status == 200:
                                                    logger.info(f"Transcription data sent to Next API for {filename}")
                                                else:
                                                    logger.error(f"Error sending transcription data to Next API for {filename}: {transcribe_resp.status} - {await transcribe_resp.text()}")

                                        except json.JSONDecodeError as e:
                                            logger.error(f"Error decoding Gemini JSON response for {filename}: {e}, Response text: '{gemini_text}'")
                                    except Exception as e:
                                            logger.error(f"Error generating content with Gemini for {filename}: {e}")
                                else:
                                    logger.info(f"No transcription found for {filename}.")
                            else:
                                logger.warning("Vosk model not loaded, skipping transcription.")
                    except wave.Error as e:
                        logger.error(f"Error reading audio file {temp_audio_path} for {filename}: {e}")
                else:
                    logger.warning(f"File size exceeds the limit ({max_file_size} bytes) for {filename}, skipping transcription.")
            else:
                logger.info(f"User {user_id} is not on the 'PRO' plan, skipping transcription for {filename}.")

        except Exception as e:
            logger.error(f"Error processing video for {filename}: {e}")
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

            # Cleanup temporary files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                logger.info(f"Deleted temporary video file: {temp_video_path}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.info(f"Deleted temporary audio file: {temp_audio_path}")
            if sid in recorded_chunks:
                del recorded_chunks[sid]

    asyncio.run(process())

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
