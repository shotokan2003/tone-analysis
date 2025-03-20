import streamlit as st
import speech_recognition as sr
import librosa
import numpy as np
import os
import warnings
import requests
from groq import Groq

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Initialize Groq client with API key from environment variable
api_key = os.environ.get("GROQ_API_KEY", "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")
groq_client = Groq(api_key=api_key)

# Verify Groq API connection
def verify_groq_connection():
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error connecting to Groq API: {str(e)}")
        return False

st.set_page_config(page_title="Speech Feedback App", page_icon="🎤", layout="wide")

@st.cache_resource
def load_models():
    recognizer = sr.Recognizer()
    return recognizer

def analyze_audio(uploaded_file):
    """Process audio file and return transcription and pitch"""
    temp_path = "temp_audio.wav"
    try:
        # Save uploaded file
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Extract audio features using librosa
        y, sr_rate = librosa.load(temp_path)
        
        # Calculate pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)
        pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.7])

        # Perform transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        return transcription, pitch

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def generate_feedback(transcription, pitch):
    if not verify_groq_connection():
        return "Unable to generate feedback due to API connection issues. Please try again later."

    prompt = f"""Analyze the following speech transcription and voice metrics for confidence, clarity, and fluency. 
    Provide specific and concise suggestions for improvement.
    
    Speech Details:
    - Transcription: {transcription}
    - Average Pitch: {pitch:.2f} Hz
    
    Please provide feedback focusing on:
    1. Voice quality and pitch modulation (considering the average pitch of {pitch:.2f} Hz)
    2. Content clarity and structure
    3. Overall delivery effectiveness
    4. Specific areas for improvement
    
    Provide the feedback in a structured, easy-to-read format."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Updated model name
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional speech coach providing constructive feedback. Be specific, encouraging, and actionable in your recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=400,
            top_p=0.95,
            stream=False
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating feedback: {str(e)}")
        return "Unable to generate feedback at this time. Please try again later."

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
        with st.spinner("Processing your audio and generating feedback..."):
            transcription, pitch = analyze_audio(uploaded_file)
            
            if transcription and pitch:
                # Display Transcription
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.subheader("📝 Transcription")
                st.info(transcription)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("🎵 Average Pitch", f"{pitch:.2f} Hz")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("📝 Word Count", str(len(transcription.split())))
                    st.markdown('</div>', unsafe_allow_html=True)

                # Generate and Display AI Feedback
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.subheader("🤖 AI Speech Coach Feedback")
                with st.spinner("Generating detailed feedback..."):
                    feedback = generate_feedback(transcription, pitch)
                    st.markdown(feedback)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Please upload a valid audio file containing clear speech.")

if __name__ == '__main__':
    main()
