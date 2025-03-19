import streamlit as st
import speech_recognition as sr
import librosa
import numpy as np
import os
import warnings
from groq import Groq

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Initialize Groq client
groq_client = Groq(
    api_key="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"  # Store your API key in Streamlit secrets
)

st.set_page_config(page_title="Speech Feedback App", page_icon="🎤", layout="wide")

@st.cache_resource
def load_models():
    recognizer = sr.Recognizer()
    return recognizer

def analyze_audio(uploaded_file):
    """Process audio file and return transcription, pitch, and speech rate"""
    temp_path = "temp_audio.wav"
    try:
        # Save uploaded file
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Extract audio features
        y, sr_rate = librosa.load(temp_path, sr=None)
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
        speech_rate = librosa.beat.tempo(y, sr=sr_rate)[0]

        # Perform transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        return transcription, pitch, speech_rate
    
    except Exception as e:
        raise Exception(f"Audio processing failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def generate_feedback(transcription, pitch, speech_rate):
    prompt = f"""Analyze the following speech transcription for confidence, clarity, and fluency. 
    Provide specific and concise suggestions for improvement.
    
    Speech Details:
    - Transcription: {transcription}
    - Pitch: {pitch:.2f} Hz
    - Speech Rate: {speech_rate:.2f} BPM
    
    Please provide feedback focusing on:
    1. Voice quality and pitch modulation
    2. Speaking pace and rhythm
    3. Clarity of pronunciation
    4. Overall delivery effectiveness
    
    Feedback:"""

    completion = groq_client.chat.completions.create(
        model="llama2-70b-3.5",
        messages=[
            {
                "role": "system",
                "content": "You are a professional speech coach providing constructive feedback."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=300,
        top_p=0.95,
        stream=False
    )
    
    return completion.choices[0].message.content

def main():
    st.markdown("""
        <style>
        .main-header {
            color: #00ff95;
            text-align: center;
            padding: 2rem 0;
        }
        .metric-container {
            background-color: #2a2a2a;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #333;
        }
        .feedback-container {
            background-color: #2a2a2a;
            padding: 2rem;
            border-radius: 10px;
            border: 1px solid #333;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🎤 Speech Analysis & Feedback</h1>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])

    if uploaded_file:
        with st.spinner("Analyzing your speech..."):
            try:
                transcription, pitch, speech_rate = analyze_audio(uploaded_file)

                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.subheader("📝 Transcription")
                st.info(transcription)
                st.markdown('</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                metrics = [
                    ("🎵 Average Pitch", f"{pitch:.2f} Hz"),
                    ("⚡ Speech Rate", f"{speech_rate:.2f} BPM"),
                    ("📝 Word Count", str(len(transcription.split())))
                ]
                
                for col, (label, value) in zip([col1, col2, col3], metrics):
                    with col:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric(label, value)
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.subheader("📈 Feedback for Improvement")
                feedback = generate_feedback(transcription, pitch, speech_rate)
                st.success(feedback)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please upload a valid WAV audio file containing clear speech.")

if __name__ == '__main__':
    main()
