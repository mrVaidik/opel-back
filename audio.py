import os
import ffmpeg
from faster_whisper import WhisperModel

# STEP 1: Set FFmpeg path manually (adjust if needed)
project_ffmpeg_path = os.path.abspath("ffmpeg-2025-03-31-git-35c091f4b7-essentials_build/bin")
os.environ["PATH"] += os.pathsep + project_ffmpeg_path

# STEP 2: Extract audio from video
def extract_audio(video_path, output_audio_path="output.wav"):
    try:
        ffmpeg.input(video_path).output(output_audio_path, ac=1, ar=16000).run(overwrite_output=True)
        print(f"✅ Audio extracted to {output_audio_path}")
        return output_audio_path
    except ffmpeg.Error as e:
        print("❌ Error during audio extraction:", e)
        return None

# STEP 3: Transcribe audio
def transcribe_audio(audio_path, model_size="small"):
    print(f"🔄 Loading Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cpu")
    print("✅ Model loaded. Starting transcription...")

    segments, _ = model.transcribe(audio_path)
    full_text = ""

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text.strip() + " "

    return full_text.strip()

# STEP 4: Video to text
def video_to_text(video_path, model_size="small"):
    if not os.path.isfile(video_path):
        print("❌ Video file does not exist.")
        return

    audio_path = extract_audio(video_path)
    if not audio_path:
        return

    transcription = transcribe_audio(audio_path, model_size)
    print("\n📝 Full Transcription:\n")
    print(transcription)
    return transcription

# STEP 5: Main
if __name__ == "__main__":
    video_path = "temp_upload/7926a21c-5e83-413d-8e32-59de0fc4604d-33e5ac2e.webm"  # replace with your actual video file
    video_to_text(video_path)
