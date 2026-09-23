# Whisper Speech-to-Text OOP Project

A clean, object-oriented Python application that leverages **OpenAI Whisper**, **SoundDevice**, and **SciPy** to transcribe speech either live from your microphone or from existing audio files, with built-in export support for `.txt` files.

---

## Features

- **Object-Oriented Design**: Encapsulates configuration and pipeline steps cleanly inside a Python class (`WhisperTranscriber`), with modules dynamically imported and stored as instance attributes.
- **Microphone Recording**: Record audio directly through your system's default microphone for a set duration.
- **File Transcription**: Support for transcribing various audio formats (`.mp3`, `.wav`, `.m4a`, `.flac`, etc.) using Whisper's underlying FFmpeg engine.
- **Text Export**: Automatically save your transcription results into a clean, UTF-8 encoded text file.

---

## Prerequisites & Installation

Make sure you have Python installed (Python 3.8+ recommended) and **FFmpeg** installed on your system path.

1. **Clone or download the project files.**
2. **Install the required dependencies:**

```bash
pip install openai-whisper sounddevice scipy numpy
```

---

## Usage Guide

Here is how you can use the `WhisperTranscriber` class in your Python scripts:

### 1. Transcribing an Existing Audio File & Saving to TXT

```python
from main import WhisperTranscriber

# Initialize transcriber (uses 'base' model by default)
transcriber = WhisperTranscriber(model_size="base")

audio_file = "my_audio.mp3"
output_txt = "transcript.txt"

try:
  text = transcriber.transcribe_file(audio_file, output_txt=output_txt)
  print("Transcription:\n", text)
except FileNotFoundError as e:
  print(e)
```

### 2. Recording from Microphone & Saving to TXT

```python
from main import WhisperTranscriber

# Initialize with a 5-second recording duration
transcriber = WhisperTranscriber(model_size="base", duration=5)

# Record, transcribe, and save automatically
transcriber.run_from_microphone(output_txt="spoken_notes.txt")
```

---

## Project Structure

- `main.py`: Contains the `WhisperTranscriber` class logic, configuration properties, and execution pipelines.
- `temp_recording.wav`: Temporary audio file created dynamically when recording from the microphone.

---

## License

This project is open-source and available under the MIT License.