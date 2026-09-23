from modules.voicemanager import WhisperTranscriber

if __name__ == "__main__":
  # Initialize the transcriber
  transcriber = WhisperTranscriber(model_size="base", duration=5)

  # --- Example 1: Record from Microphone & Save to TXT ---
  # transcriber.run_from_microphone(output_txt="my_spoken_notes.txt")

  # --- Example 2: Transcribe an Existing File & Save to TXT ---
  audio_file_path = "audio/temp_recording.wav"  # Replace with your audio file path
  output_text_path = "text/file_transcription.txt"

  try:
    result = transcriber.transcribe_file(file_path=audio_file_path, output_txt=output_text_path)
    print("\n--- File Transcription Result ---")
    print(result)
  except FileNotFoundError as e:
    print(e)