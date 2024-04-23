import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import tempfile
import os
from tensorflow.keras.models import load_model

# Define your preprocessing functions
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess1(file_path):
    # Load the WAV file and convert it to a tensor
    wav = load_wav_16k_mono(file_path)
    # Trim the audio to 25000 samples
    wav = wav[:25000]
    # Add zero padding to make the length 25000
    zero_padding = tf.zeros([25000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    # Compute the Short-Time Fourier Transform (STFT) of the audio
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    # Compute the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)
    # Add a channel dimension to the spectrogram
    spectrogram = tf.expand_dims(spectrogram, axis=0)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram

# Load the model
with open("/content/drive/MyDrive/arc/model.pkl", 'rb') as f:
        model = pickle.load(f)

# Define the Streamlit app
def main():
    st.title('Audio classification')
    # Load the model
    with open("/content/drive/MyDrive/arc/label_encoder.pkl", 'rb') as f:
        label_encoder= pickle.load(f)

    # File uploader component
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # If a file has been uploaded, read its contents
        audio_bytes = uploaded_file.read()
        print("Audio Bytes:", audio_bytes)  # Debugging statement
        if audio_bytes is not None:
            st.audio(audio_bytes, format='audio/mp3')
        else:
            st.write("Failed to read audio file.")
        
        # Write the file contents to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "uploaded_audio.wav")
        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)

        # Preprocess the uploaded audio file
        processed_audio = preprocess1(temp_file_path)
        # Make prediction using the loaded model
        prediction = model.predict(processed_audio)
        prediction = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        # Display prediction
        st.write('Prediction:', prediction)

# Run the Streamlit app
if __name__ == "__main__":
    main()
