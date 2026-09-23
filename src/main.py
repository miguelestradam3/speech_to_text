from modules.voicemanager import WhisperTranscriber

# --- EXAMPLE 1: Transcribe an Existing Audio File ---
# Replace with the path to your audio file (supports .mp3, .wav, .m4a, etc.)
audio_file_path = "audio/temp_recording.wav"  # Change this to your audio file path

if __name__ == "__main__":
  # Initialize the transcriber
  transcriber = WhisperTranscriber(model_size="base")

  try:
    result = transcriber.transcribe_file(audio_file_path)
    print("\n--- File Transcription Result ---")
    print(result)
  except FileNotFoundError as e:
    print(e)

  # --- EXAMPLE 2: Record from Microphone ---
  transcriber.run_from_microphone()