"""Record an audio or playback a wave file using PyAudio."""
# https://people.csail.mit.edu/hubert/pyaudio/#examples

import wave
import sys
import time
import argparse

import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 5

def record_audio(file):
    
    with wave.open(file, 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        print('Recording...')
        for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
            wf.writeframes(stream.read(CHUNK))
        print('Done')

        stream.close()
        p.terminate()
        
    return f'Recorded audio at {file}'
    
def playback_audio(file):
    with wave.open(file, 'rb') as wf:
        # Define callback for playback (1)
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            # If len(data) is less than requested frame_count, PyAudio automatically
            # assumes the stream is finished, and the stream stops.
            return (data, pyaudio.paContinue)

        # Instantiate PyAudio and initialize PortAudio system resources (2)
        p = pyaudio.PyAudio()

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


def main():
    parser = argparse.ArgumentParser(description="Record or playback audio.")
    parser.add_argument('mode', choices=['record', 'playback'], 
                        help="Mode to run: 'record' or 'playback'.")
    parser.add_argument('file', type=str, help="File to record to or playback from.")

    args = parser.parse_args()

    if args.mode == 'record':
        record_audio(args.file)
    elif args.mode == 'playback':
        playback_audio(args.file)

if __name__ == '__main__':
    main()
