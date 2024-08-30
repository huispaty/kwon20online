import pyaudio
import wave
from pydub import AudioSegment
from pydub.utils import make_chunks
import sys
import numpy as np
import librosa
import queue

CHANNELS = 2 if sys.platform != 'darwin' else 1
RATE = 16000
CHUNK = 512

class AudioFileStream: # librosa
    def __init__(self, file_path, channels, rate, chunk_size):
        
        # directly decode audio
        self._audio_decoded, _ = librosa.load(file_path, sr=rate, mono=True)
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=channels,
                                  rate=rate,
                                  input=True)
        
        # convert audio file into chunks
        self._buff = self._make_chunks(chunk_size)
    
    def _make_chunks(self, chunk_size):
        chunks = queue.Queue()
        for i in range(0, len(self._audio_decoded), chunk_size):
            chunk = self._audio_decoded[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunks.put(chunk)
        return chunks
    
    def generator(self):
        while not self._buff.empty():
            chunk = self._buff.get()
            if chunk is None:
                return 
            else: 
                yield chunk
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
      