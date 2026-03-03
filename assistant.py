import os
import sys
import time
import queue
import threading
import numpy as np
import pyaudio
import ollama
from faster_whisper import WhisperModel
import pyttsx3
from rag_engine import RAGEngine

# Configuration
MODEL_NAME = "phi3:latest"
WHISPER_MODEL_SIZE = "base.en" 
WAKE_WORD = "assistant"

class OfflineAssistant:
    def __init__(self, callback=None):
        self.callback = callback
        print("Initializing Offline AI Assistant...")

        # Initialize TTS Engine
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        # Safety check for voice selection
        if len(self.voices) > 1:
            self.engine.setProperty('voice', self.voices[1].id) 

        self.engine.setProperty('rate', 180)

        # Initialize STT (Faster-Whisper)
        print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
        # Run on CPU with int8 quantization for speed
        self.stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

        # Initialize RAG Engine
        print("Initializing RAG Engine (Local Vector DB)...")
        self.rag = RAGEngine()
        self.rag.load_db()

        # Audio constants
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.p = pyaudio.PyAudio()

        # State Management
        self.is_speaking = False
        self.stop_event = threading.Event()

    def speak(self, text):
        """Convert text to speech without locking the main thread permanently."""
        if not text: return
        print(f"Assistant: {text}")
        if self.callback:
            self.callback("assistant", text)

        self.is_speaking = True
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        finally:
            self.is_speaking = False

    def get_ollama_response(self, prompt):
        """Get response from Ollama with RAG context and speak as it streams."""
        try:
            # 1. Retrieve Context from Vector DB
            context = self.rag.query(prompt)

            # 2. Build the System Prompt with Context
            system_prompt = (
                "You are a helpful offline personal AI assistant. "
                "Use the following pieces of retrieved context from the user's personal documents "
                "to answer the question. If you don't know the answer or it's not in the context, "
                "you can still answer based on your general knowledge but mention if you are "
                "using general knowledge vs personal context.\n\n"
                f"PERSONAL CONTEXT:\n{context}\n\n"
                "Be concise and friendly."
            )

            response = ollama.chat(
                model=MODEL_NAME, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                stream=True
            )

            full_content = ""
            sentence_buffer = ""
            
            for chunk in response:
                if self.stop_event.is_set():
                    break
                
                content = chunk['message']['content']
                full_content += content
                sentence_buffer += content

                # If we have a sentence-ending punctuation, speak it
                if any(punct in content for punct in ['.', '!', '?', '\n']):
                    if sentence_buffer.strip():
                        self.speak(sentence_buffer.strip())
                        sentence_buffer = ""

            # Speak any remaining text
            if sentence_buffer.strip():
                self.speak(sentence_buffer.strip())

            return full_content
        except Exception as e:
            err = f"I'm having trouble reaching the brain: {str(e)}"
            self.speak(err)
            return err

    def listen_and_transcribe(self):
        """Continuously capture audio and transcribe."""
        try:
            stream = self.p.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk_size)
        except Exception as e:
            print(f"Could not open microphone: {e}")
            if self.callback:
                self.callback("system", f"Microphone error: {e}")
            return

        print("\n--- Assistant is Listening ---")

        frames = []
        silent_chunks = 0
        speech_started = False

        try:
            while not self.stop_event.is_set():
                if self.is_speaking:
                    time.sleep(0.5)
                    continue

                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                    rms = np.sqrt(np.mean(audio_data**2))

                    if rms > 0.012: 
                        if not speech_started:
                            print("Listening...")
                            speech_started = True
                        frames.append(audio_data)
                        silent_chunks = 0
                    elif speech_started:
                        frames.append(audio_data)
                        silent_chunks += 1

                    if speech_started and silent_chunks > 18:
                        print("Processing voice...")
                        full_audio = np.concatenate(frames)

                        segments, _ = self.stt_model.transcribe(full_audio, beam_size=5)
                        text = " ".join([segment.text for segment in segments]).strip()

                        if text:
                            print(f"You: {text}")
                            if self.callback:
                                self.callback("user", text)
                            self.handle_input(text)

                        frames = []
                        speech_started = False
                        silent_chunks = 0
                except IOError:
                    pass
        finally:
            stream.stop_stream()
            stream.close()

    def handle_input(self, text):
        """Logic for acting on commands."""
        text_lower = text.lower()

        if any(cmd in text_lower for cmd in ["exit assistant", "stop assistant", "turn off assistant"]):
            self.speak("Voice assistant mode deactivated.")
            self.stop_event.set()
            return

        if "update your knowledge" in text_lower or "reindex" in text_lower:
            self.speak("Updating my memory from your personal files.")
            status = self.rag.index_data()
            self.speak(status)
            return

        self.get_ollama_response(text)

    def run(self):
        """Start the assistant loop."""
        self.stop_event.clear()
        self.listen_and_transcribe()

    def stop(self):
        self.stop_event.set()
        self.p.terminate()

if __name__ == "__main__":
    assistant = OfflineAssistant()
    assistant.run()
