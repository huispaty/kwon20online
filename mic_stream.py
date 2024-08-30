import pyaudio
import queue
import time
import numpy as np

CHUNK = 2048
RATE = 44100


class MicrophoneStream(object):
    "Microphone input class"
    def __init__(self, rate, chunk, channels):
        self._rate = rate
        self._chunk = chunk
        self._channels = channels 
        
        self._buff = queue.Queue() # microphone input buffer
        self.closed = True    
    
    def __enter__(self):
        print('___ENTER___')
        # create pyaudio interface
        self._audio_interface = pyaudio.PyAudio()
        # open microphone in 16-bit
        # use _fill_buffer as callback : call when the buffer is filled
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16, # mic streaming width = 2
            channels=self._channels, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        print('___EXIT___')
        self._audio_stream.stop_stream()
        self._audio_stream.close()

        self.closed = True 
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()
    
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        # called whenever the microphone buffer is filled (CHUNK = CHUNK)
        # put microphone input into the queue and return
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self, time_limit=None):
        start_time = time.time()
        
        while not self.closed:
            if time_limit is not None and (time.time() - start_time) > time_limit:
                print("Recording time limit reached, stop recording.")
                return
            
            # get data from buffer
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def main():
    # 마이크 열기 
    with MicrophoneStream(RATE, CHUNK, 1) as stream: 
        # create micro data handle generator
        audio_generator = stream.generator()
        for _ in range(0, RATE // CHUNK * 5): # test: get microphone data for 5 seconds
            # get data from buffer 
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            # simulate real-time streaming
            time.sleep(0.001)

if __name__ == '__main__':
    main()
