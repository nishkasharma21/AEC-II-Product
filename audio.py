from flask import Flask, Response,render_template
import pyaudio

app = Flask(__name__)
 
audio1 = pyaudio.PyAudio()

FORMAT = audio1.get_format_from_width(width=2)
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 5

def genHeader(sampleRate, bitsPerSample, channels):
    datasize = 2000*10**6
    o = bytes("RIFF",'ascii')                                               # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4,'little')                               # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE",'ascii')                                              # (4byte) File type
    o += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
    o += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
    o += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2,'little')                                    # (2byte)
    o += (sampleRate).to_bytes(4,'little')                                  # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
    o += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
    o += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
    return o

@app.route('/audio')
def audio():
    # start Recording
    def sound():

        CHUNK = 100
        sampleRate = 44100
        bitsPerSample = 16
        channels = 1
        wav_header = genHeader(sampleRate, bitsPerSample, channels)

        

        stream = audio1.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output= True, 
                        frames_per_buffer=CHUNK)
        
        output_latency = stream.get_output_latency()
        print(f"Output Latency: {output_latency} seconds")
        input_latency = stream.get_input_latency()
        print(f"Input Latency: {input_latency} seconds")

        print("recording...")
        #frames = []
        first_run = True
        while True:
           if first_run:
               data = wav_header + stream.read(CHUNK)
               first_run = False
           else:
               data = stream.read(CHUNK)
           yield(data)

    return Response(sound())

@app.route('/')
def index():
    return render_template('index_audio.html')

      
if __name__ == "__main__":
    app.run(host='localhost', debug=True, threaded=True,port=8001)