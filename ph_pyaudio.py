"""Record an audio or playback a wave file using PyAudio."""
# https://people.csail.mit.edu/hubert/pyaudio/#examples

import sys
import time
import argparse
import numpy as np

import pyaudio
import wave
import librosa
import sounddevice as sd
from pydub import AudioSegment

CHANNELS = 1 if sys.platform == 'darwin' else 2
CHUNK = 512
RATE = 16000
FORMAT = pyaudio.paInt16 # 16-bit, format = 2
RECORD_SECONDS = 5


'''
PyAudio format -- width mappings
    width = sample width in bytes
    1 byte = 8 bits = PaUInt8
    2 bytes = 16 bits = PaInt16
    3 bytes = 24 bits = PaInt24
    4 bytes = 32 bits = PaFloat32
'''


def record_audio(file):
    
    with wave.open(file, 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        print('-' * 42)
        print('Audio file params')
        print(f'channels: {wf.getnchannels()}')
        print(f'sample width: {wf.getsampwidth()}')
        print(f'frame rate: {wf.getframerate()}')
        print(f'format: {p.get_format_from_width(wf.getsampwidth())}')
        print('-' * 42)

        print('Recording...')
        for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
            wf.writeframes(stream.read(CHUNK))
        print('Done')

        stream.close()
        p.terminate()
        
    return f'Recorded audio at {file}'
    
    
def playback_audio_wave(file):
    with wave.open(file, 'rb') as wf:
        # Define callback for playback (1)
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            # If len(data) is less than requested frame_count, PyAudio automatically
            # assumes the stream is finished, and the stream stops.
            return (data, pyaudio.paContinue)

        # Instantiate PyAudio and initialize PortAudio system resources (2)
        p = pyaudio.PyAudio()
        
        print('-' * 42)
        print('Audio file params')
        print(f'format: {p.get_format_from_width(wf.getsampwidth())}')
        print(f'channels: {wf.getnchannels()}')
        print(f'frame rate: {wf.getframerate()}')
        print(f'sample width: {wf.getsampwidth()}')
        print('-' * 42)
        
        # ------------------------------------------
        # Audio file params
        # channels: 2
        # sample width: 3
        # frame rate: 44100
        # format: 4
        # ------------------------------------------
        
        # Open stream using callback (3)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        stream_callback=callback)

        # Wait for stream to finish (4)
        while stream.is_active():
            time.sleep(0.1)

        # Close the stream (5)
        stream.close()

        # Release PortAudio system resources (6)
        p.terminate()


def playback_audio_librosa_sounddevice(file):
    # Load the audio file using librosa
    y, sr = librosa.load(file, sr=RATE, mono=True)  # sr=None keeps the original sampling rate

    # Display the audio file parameters
    print('-' * 42)
    print('Audio file params')
    print(f'channels: {CHANNELS}')  # librosa loads audio as mono by default
    print(f'sample rate: {sr}')
    print(f'duration: {librosa.get_duration(y=y, sr=sr)} seconds')
    print('-' * 42)
    
    # ------------------------------------------
    # Audio file params
    # channels: 1
    # sample rate: 16000
    # duration: 6.65625 seconds
    # ------------------------------------------
    
    # Play the audio using sounddevice
    sd.play(y, sr)

    # Wait until the audio is finished playing
    sd.wait()



def playback_audio_librosa(file):
    # Load the audio file using librosa
    y, sr = librosa.load(file, sr=RATE, mono=True)

    # Display the audio file parameters
    print('-' * 42)
    print('Audio file params')
    print(f'channels: 1')  # librosa loads audio as mono by default
    print(f'sample rate: {sr}')
    print(f'duration: {librosa.get_duration(y=y, sr=sr)} seconds')
    print('-' * 42)

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open an output stream
    stream = p.open(format=pyaudio.paFloat32,  # librosa outputs float32 data
                    channels=CHANNELS,         
                    rate=sr,                   
                    output=True)
    
    # Stream the audio in chunks
    chunk_size = CHUNK  # You can adjust this size
    num_chunks = len(y) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        stream.write(y[start:end].astype(np.float32).tobytes())
        
    # pad the last chunk with silence
    remainder = len(y) % chunk_size
    if remainder > 0:
        last_chunk = np.zeros(chunk_size, dtype=np.float32)
        last_chunk[:remainder] = y[-remainder:]
        stream.write(last_chunk.tobytes())

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()


def playback_audio_pydub(file):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(file)
    audio = audio.set_channels(1)  # Convert to mono
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize to the range [-1, 1]
    samples /= np.max(np.abs(samples))

    # Display the audio file parameters
    print('-' * 42)
    print('Audio file params')
    print(f'channels: 1')  # Audio is now mono
    print(f'sample rate: {sr}')
    print(f'duration: {len(samples) / sr} seconds')
    print('-' * 42)

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open an output stream
    stream = p.open(format=pyaudio.paFloat32,  # pydub outputs float32 data
                    channels=1,                # Mono audio
                    rate=sr,                   # Sample rate
                    output=True)
    
    # Stream the audio in chunks
    chunk_size = 1024  # You can adjust this size
    num_chunks = len(samples) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        stream.write(samples[start:end].tobytes())
        
    # Pad the last chunk with silence
    remainder = len(samples) % chunk_size
    if remainder > 0:
        last_chunk = np.zeros(chunk_size, dtype=np.float32)
        last_chunk[:remainder] = samples[-remainder:]
        stream.write(last_chunk.tobytes())

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()




def main():
    parser = argparse.ArgumentParser(description="Record or playback audio.")
    parser.add_argument('mode', choices=['record', 'playback'], 
                        help="Mode to run: 'record' or 'playback'.")
    parser.add_argument('file', type=str, help="File to record to or playback from.")

    args = parser.parse_args()

    if args.mode == 'record':
        record_audio(args.file)
    elif args.mode == 'playback':
        # playback_audio_wave(args.file)
        # playback_audio_librosa_sounddevice(args.file)
        playback_audio_librosa(args.file)
        # playback_audio_pydub(args.file)
        

if __name__ == '__main__':
    main()
