import streamlit as st
import speech_recognition as sr
import librosa
import numpy as np
import os
import warnings
import requests
from groq import Groq
from pydub import AudioSegment
import tempfile

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Initialize Groq client with API key from environment variable
api_key = os.environ.get("GROQ_API_KEY", "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")
groq_client = Groq(api_key=api_key)

# Supported audio formats
SUPPORTED_FORMATS = {
    'wav': 'WAV',
    'mp3': 'MP3',
    'mp4': 'MP4',
    'm4a': 'M4A',
    'ogg': 'OGG',
    'flac': 'FLAC'
}

def convert_to_wav(audio_file, file_type):
    """Convert audio file to WAV format"""
    try:
        # Create a temporary file for the converted audio
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, "temp_audio.wav")
        
        # Load the audio file using pydub
        audio = AudioSegment.from_file(audio_file, format=file_type.lower())
        
        # Export as WAV
        audio.export(temp_wav, format="wav")
        return temp_wav
    except Exception as e:
        st.error(f"Error converting audio file: {str(e)}")
        return None

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
    """Process audio file and return transcription, pitch details, and pause information"""
    try:
        # Get file extension and convert to WAV if needed
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in SUPPORTED_FORMATS:
            st.error(f"Unsupported file format. Please upload one of: {', '.join(SUPPORTED_FORMATS.keys())}")
            return None, None

        # Create a temporary file for the uploaded audio
        temp_dir = tempfile.gettempdir()
        temp_input = os.path.join(temp_dir, f"input_audio.{file_extension}")
        
        # Save uploaded file
        with open(temp_input, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Convert to WAV if needed
        if file_extension != 'wav':
            temp_wav = convert_to_wav(temp_input, file_extension)
            if not temp_wav:
                return None, None
        else:
            temp_wav = temp_input

        # Extract audio features using librosa
        y, sr_rate = librosa.load(temp_wav)
        
        # Detect silence/pauses
        intervals = librosa.effects.split(y, top_db=30)
        
        # Calculate duration of pauses
        pauses = []
        for i in range(len(intervals)-1):
            pause_duration = (intervals[i+1][0] - intervals[i][1]) / sr_rate
            if pause_duration > 0.3:  # Only count pauses longer than 0.3 seconds
                pauses.append(pause_duration)

        # Perform transcription with timestamps
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        # Split audio into segments based on pauses
        segments = []
        segment_pitches = []
        
        for i in range(len(intervals)):
            start_sample, end_sample = intervals[i]
            segment = y[start_sample:end_sample]
            
            # Calculate pitch for each segment
            if len(segment) > 0:
                pitches, magnitudes = librosa.piptrack(y=segment, sr=sr_rate)
                segment_pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.7])
                segment_pitches.append(segment_pitch)
                
                # Convert samples to time
                start_time = float(start_sample) / sr_rate
                end_time = float(end_sample) / sr_rate
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'pitch': segment_pitch
                })

        # Calculate average pitch for reference
        avg_pitch = np.mean([s['pitch'] for s in segments])

        speech_metrics = {
            'segments': segments,
            'pauses': pauses,
            'average_pitch': avg_pitch,
            'pause_count': len(pauses),
            'average_pause_duration': np.mean(pauses) if pauses else 0
        }

        return transcription, speech_metrics

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None
    
    finally:
        # Clean up temporary files
        for temp_file in [temp_input, temp_wav]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def generate_feedback(transcription, speech_metrics):
    if not verify_groq_connection():
        return "Unable to generate feedback due to API connection issues. Please try again later."

    # Create a detailed analysis of speaking patterns
    pause_analysis = f"""
    - Number of significant pauses: {speech_metrics['pause_count']}
    - Average pause duration: {speech_metrics['average_pause_duration']:.2f} seconds
    """

    # Analyze pitch variations
    pitch_variations = []
    for i, segment in enumerate(speech_metrics['segments']):
        variation = ((segment['pitch'] - speech_metrics['average_pitch']) / speech_metrics['average_pitch']) * 100
        pitch_variations.append(variation)

    prompt = f"""Analyze the following speech transcription and detailed voice metrics for confidence, clarity, and fluency. 
    Provide specific and concise suggestions for improvement.
    
    Speech Details:
    - Transcription: {transcription}
    - Average Pitch: {speech_metrics['average_pitch']:.2f} Hz
    
    Speaking Pattern Analysis:
    {pause_analysis}
    - Pitch variation between segments: Ranges from {min(pitch_variations):.1f}% to {max(pitch_variations):.1f}% from average
    
    Please provide feedback focusing on:
    1. Voice quality and pitch modulation patterns
    2. Effective use of pauses and pacing
    3. Content clarity and structure
    4. Overall delivery effectiveness
    5. Specific areas for improvement
    
    Consider how the speaker's use of pauses affects their message delivery and how pitch variations contribute to engagement.
    Provide the feedback in a structured, easy-to-read format."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced professional speech coach providing concise, structured, and actionable feedback. Your responses must be brief, specific, and supportive. Focus explicitly on the speaker's voice modulation, pitch consistency, use of pauses, content clarity, and overall delivery effectiveness. Provide clear recommendations for improvement, formatted into distinct bullet points, without unnecessary elaboration."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500,
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

    # File uploader with supported formats
    uploaded_file = st.file_uploader(
        f"Upload an audio file (Supported formats: {', '.join(SUPPORTED_FORMATS.keys())})", 
        type=list(SUPPORTED_FORMATS.keys())
    )

    if uploaded_file:
        with st.spinner("Processing your audio and generating feedback..."):
            transcription, speech_metrics = analyze_audio(uploaded_file)
            
            if transcription and speech_metrics:
                # Display Transcription
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.subheader("📝 Transcription")
                st.info(transcription)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("🎵 Average Pitch", f"{speech_metrics['average_pitch']:.2f} Hz")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("⏱️ Pauses", f"{speech_metrics['pause_count']}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("📝 Word Count", str(len(transcription.split())))
                    st.markdown('</div>', unsafe_allow_html=True)

                # Display Pitch Variation Graph
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.subheader("📊 Pitch Variation Analysis")
                pitch_data = [segment['pitch'] for segment in speech_metrics['segments']]
                st.line_chart(pitch_data)
                st.markdown('</div>', unsafe_allow_html=True)

                # Generate and Display AI Feedback
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.subheader("🤖 AI Speech Coach Feedback")
                with st.spinner("Generating detailed feedback..."):
                    feedback = generate_feedback(transcription, speech_metrics)
                    st.markdown(feedback)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Please upload a valid audio file containing clear speech.")

if __name__ == '__main__':
    main()
