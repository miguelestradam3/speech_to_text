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

  def transcribe_file(self, file_path: str) -> str:
    """Transcribes any specified audio file using the whisper attribute."""
    self.load_model()

    if not self.os.path.exists(file_path):
      raise FileNotFoundError(f"Could not find audio file at: {file_path}")

    print(f"🔄 Transcribing file: {file_path}...")
    result = self.model.transcribe(file_path, fp16=False)
    return result["text"].strip()

  def run_from_microphone(self):
    """Pipeline to record from the microphone and transcribe the result."""
    self.load_model()
    self.record_audio()
    self.save_audio()

    transcription = self.transcribe_file(self.temp_filename)
    print("\n--- Microphone Transcription Result ---")
    print(transcription)