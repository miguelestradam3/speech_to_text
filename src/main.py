import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import whisper

# Configuration
DURATION = 5  # Recording duration in seconds
SAMPLE_RATE = 16000  # Whisper natively expects 16kHz audio
TEMP_FILENAME = "temp_recording.wav"


def main():
  # 1. Load the Whisper model (options: 'tiny', 'base', 'small', 'medium', 'large')
  print("Loading Whisper model ('base')...")
  model = whisper.load_model("base")

  # 2. Record audio from the microphone
  print(f"🎤 Recording for {DURATION} seconds... Speak now!")
  audio_data = sd.rec(
      int(DURATION * SAMPLE_RATE),
      samplerate=SAMPLE_RATE,
      channels=1,
      dtype=np.int16,
  )
  sd.wait()  # Wait until the recording is finished
  print("⏹️ Recording finished.")

  # 3. Save the recorded audio using SciPy
  wav.write(TEMP_FILENAME, SAMPLE_RATE, audio_data)

  # 4. Transcribe the audio file with Whisper
  print("🔄 Transcribing audio...")
  result = model.transcribe(TEMP_FILENAME, fp16=False)

  # 5. Output the result
  print("\n--- Transcription Result ---")
  print(result["text"].strip())


if __name__ == "__main__":
  main()