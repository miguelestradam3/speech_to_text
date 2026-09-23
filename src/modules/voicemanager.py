class WhisperTranscriber:

  def __init__(
      self,
      model_size: str = "base",
      duration: int = 5,
      sample_rate: int = 16000,
      temp_filename: str = "temp_recording.wav",
  ):
    """Initializes configuration and imports required modules as object attributes."""
    # Import modules inside __init__ and bind them to self
    import os
    import numpy as np
    import scipy.io.wavfile as wav
    import sounddevice as sd
    import whisper

    self.os = os
    self.np = np
    self.wav = wav
    self.sd = sd
    self.whisper = whisper

    self.model_size = model_size
    self.duration = duration
    self.sample_rate = sample_rate
    self.temp_filename = temp_filename
    self.model = None
    self.audio_data = None

  def load_model(self):
    """Loads the Whisper model into memory if not already loaded."""
    if self.model is None:
      print(f"Loading Whisper model ('{self.model_size}')...")
      self.model = self.whisper.load_model(self.model_size)

  def record_audio(self):
    """Records audio from the microphone using the sounddevice attribute."""
    print(f"🎤 Recording for {self.duration} seconds... Speak now!")
    self.audio_data = self.sd.rec(
        int(self.duration * self.sample_rate),
        samplerate=self.sample_rate,
        channels=1,
        dtype=self.np.int16,
    )
    self.sd.wait()  # Wait until recording finishes
    print("⏹️ Recording finished.")

  def save_audio(self):
    """Saves the recorded audio data as a WAV file using the scipy wav attribute."""
    if self.audio_data is None:
      raise ValueError("No audio data found. Please record audio first.")
    self.wav.write(self.temp_filename, self.sample_rate, self.audio_data)

  def save_text_to_file(self, text: str, output_path: str = "transcription.txt"):
    """Saves the given transcription text to a .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(text)
    print(f"💾 Transcription saved successfully to: {output_path}")

  def transcribe_file(self, file_path: str, output_txt: str = None) -> str:
    """Transcribes an audio file and optionally saves the text to a .txt file."""
    self.load_model()

    if not self.os.path.exists(file_path):
      raise FileNotFoundError(f"Could not find audio file at: {file_path}")

    print(f"🔄 Transcribing file: {file_path}...")
    result = self.model.transcribe(file_path, fp16=False)
    transcription = result["text"].strip()

    # Save to text file if an output path is provided
    if output_txt:
      self.save_text_to_file(text=transcription, output_path=output_txt)

    return transcription

  def run_from_microphone(self, output_txt: str = "mic_transcription.txt"):
    """Pipeline to record from the microphone, transcribe, and save to a text file."""
    self.load_model()
    self.record_audio()
    self.save_audio()

    transcription = self.transcribe_file(file_path=self.temp_filename, output_txt=output_txt)
    print("\n--- Microphone Transcription Result ---")
    print(transcription)