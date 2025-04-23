import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

#model = "piper-voices/en/ryan/high/en_US-ryan-high.onnx"
# model = "piper-voices/en/ryan/low/en_US-ryan-low.onnx"

# model = "piper-voices/en/ryan/medium/en_US-ryan-medium.onnx"

# model = "piper-voices/en/hfc_female/medium/en_US-hfc_female-medium.onnx"
# model = "piper-voices/en/ljspeech/high/en_US-ljspeech-high.onnx"
model = "piper-voices/en/ljspeech/medium/en_US-ljspeech-medium.onnx"

voice = PiperVoice.load(model)
text = "Your excellency Dean R. Thompson, Welcome to U-S-Y-C 2025. You are looking fantastic and ready to shine."

# Setup a sounddevice OutputStream with appropriate parameters
# The sample rate and channels should match the properties of the PCM data
stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
stream.start()

for audio_bytes in voice.synthesize_stream_raw(text):
    int_data = np.frombuffer(audio_bytes, dtype=np.int16)
    stream.write(int_data)

stream.stop()
stream.close()