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
import google.generativeai as genai
import PyPDF2
from dotenv import load_dotenv
import re

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
load_dotenv()

# Initialize Groq client with API key from environment variable
api_key = os.environ.get("GROQ_API_KEY", "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")
groq_client = Groq(api_key=api_key)

# Initialize Gemini
def initialize_gemini():
    google_api_key = "AIzaSyAjeaMnL97sqU-IZbjwho65DTDjMtkjlF4"
    genai.configure(api_key=google_api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# Function to remove personal information
def remove_details(text):
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL REDACTED]", text)
    text = re.sub(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", "[PHONE REDACTED]", text)
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        clean_text = remove_details(text.strip())
        
        # Check if PDF content is too short or empty
        if len(clean_text) < 50:
            return None, "The uploaded PDF appears to be empty or contains very little text. Please upload a valid resume."
            
        return clean_text, None
    except Exception as e:
        return None, f"Error extracting text from PDF: {str(e)}"

# Function to check for inappropriate content
def contains_inappropriate_content(text):
    # List of words/patterns that might indicate hate speech or inappropriate content
    # This is a simplified check - in production, you might use a more sophisticated content moderation API
    inappropriate_patterns = [
        r'\b(hate|hating|hateful)\b', 
        r'\b(racist|racism|racial slur)\b',
        r'\b(sexist|sexism)\b',
        r'\b(offensive|vulgar|explicit)\b',
        r'\bslur\b',
        # Add more patterns as needed
    ]
    
    text_lower = text.lower()
    for pattern in inappropriate_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

# Function to generate interview questions
def generate_questions(resume_text, num_questions, job_description=None):
    model = initialize_gemini()
    
    # Check for inappropriate content
    if contains_inappropriate_content(resume_text):
        return [], "The resume contains potentially inappropriate content. Please review and upload an appropriate resume."
    
    if job_description and contains_inappropriate_content(job_description):
        return [], "The job description contains potentially inappropriate content. Please review and provide appropriate content."
    
    # Extract candidate name from resume for personalization
    name_match = re.search(r"(?i)name[:\s]+([A-Za-z\s]+)", resume_text)
    candidate_name = name_match.group(1).strip() if name_match else "the candidate"
    
    if job_description:
        prompt = f"""Based on this resume and job description, generate exactly {num_questions} relevant interview questions.
        Focus on technical skills, practical knowledge, and experience mentioned in the resume that align with the job requirements.
        Give questions as a numbered list from 1 to {num_questions}. Make questions specific to {candidate_name}'s background and the job requirements.
        Do not include any introduction or additional text beyond the numbered questions.
        Resume extracted Text: {resume_text}
        
        Job Description: {job_description}"""
    else:
        prompt = f"""Based on this resume, generate exactly {num_questions} relevant interview questions.
        Focus on technical skills, practical knowledge, and experience mentioned in the resume.
        Give questions as a numbered list from 1 to {num_questions}. Make questions specific to {candidate_name}'s background.
        Do not include any introduction or additional text beyond the numbered questions.
        Resume extracted Text: {resume_text}"""

    try:
        response = model.generate_content(prompt)
        questions = response.text.strip().split('\n')
        
        # Clean and validate questions
        valid_questions = []
        for q in questions:
            q = q.strip()
            if not q:
                continue
            
            # Remove numbering and any prefixes
            if re.match(r'^\d+[\.\)]\s+', q):
                q = re.sub(r'^\d+[\.\)]\s+', '', q)
            
            # Validate that this is an actual question (not an introduction or explanation)
            if len(q) > 15 and ('?' in q or re.search(r'\b(explain|describe|discuss|tell|how|what|when|where|why|which|who)\b', q, re.IGNORECASE)):
                # Check for inappropriate content in the question
                if not contains_inappropriate_content(q):
                    valid_questions.append(q)
        
        # Ensure we have exactly the requested number of questions
        if len(valid_questions) > num_questions:
            valid_questions = valid_questions[:num_questions]
        
        # If we don't have enough valid questions, generate more
        if len(valid_questions) < num_questions and len(valid_questions) > 0:
            additional_needed = num_questions - len(valid_questions)
            retry_prompt = f"""Based on this resume, generate exactly {additional_needed} more interview questions.
            Questions should be different from these already generated: {valid_questions}
            Focus on technical skills and experience in the resume.
            Give only the questions without numbering or other text.
            Resume: {resume_text}"""
            
            retry_response = model.generate_content(retry_prompt)
            additional_questions = [q.strip() for q in retry_response.text.strip().split('\n') if q.strip()]
            
            # Clean and validate additional questions
            for q in additional_questions:
                if len(valid_questions) >= num_questions:
                    break
                    
                if re.match(r'^\d+[\.\)]\s+', q):
                    q = re.sub(r'^\d+[\.\)]\s+', '', q)
                    
                if len(q) > 15 and ('?' in q or re.search(r'\b(explain|describe|discuss|tell|how|what|when|where|why|which|who)\b', q, re.IGNORECASE)):
                    if not contains_inappropriate_content(q):
                        valid_questions.append(q)
        
        return valid_questions, None
    except Exception as e:
        return [], f"Error generating questions: {str(e)}"

# Function to analyze answer content and provide feedback with job relevance
def analyze_answer_content(audio_file_path, question, job_description=None, resume_text=None):
    model = initialize_gemini()
    
    try:
        # Check if the audio file exists and has content
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # If file is smaller than 1KB, likely empty or corrupted
            return "The audio file appears to be empty or too short to analyze. Score: 0/100"
        
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        
        # Extract candidate name from resume
        candidate_name = "the candidate"
        if resume_text:
            name_match = re.search(r"(?i)name[:\s]+([A-Za-z\s]+)", resume_text)
            if name_match:
                candidate_name = name_match.group(1).strip()
        
        if job_description:
            prompt = f"""Analyze this answer for the question: '{question}'.
            The candidate's name is {candidate_name}.
            Consider the following job description: '{job_description}'.
            
            Provide feedback on:
            1. Correctness and depth of the answer
            2. Relevance to the job requirements
            3. Communication clarity
            
            Include a job fit score (0-100%) indicating how well the answer aligns with what employers would look for based on the job description.
            If you detect no speech, very unclear speech, or if the audio appears to be empty or inappropriate, give a score of 0% and note the issue.
            
            Format as a professional, structured assessment with clear ratings and actionable improvement suggestions.
            IMPORTANT: Always refer to the candidate as {candidate_name} throughout your analysis.
            Check for any inappropriate content in the speech and flag it if detected.
            """
        else:
            prompt = f"""Analyze this answer for the question: '{question}'.
            The candidate's name is {candidate_name}.
            
            Provide feedback on:
            1. Correctness and depth of the answer
            2. Communication clarity and structure
            
            Include ratings and provide suggestions for improvement. Analyze as you are a technical interviewer and provide feedback accordingly.
            If you detect no speech, very unclear speech, or if the audio appears to be empty or inappropriate, give a score of 0% and note the issue.
            
            IMPORTANT: Always refer to the candidate as {candidate_name} throughout your analysis.
            Check for any inappropriate content in the speech and flag it if detected.
            """
        
        response = model.generate_content([
            {"mime_type": "audio/wav", "data": audio_data},
            prompt
        ])
        
        content = response.text
        
        # Check for inappropriate content in the response
        if contains_inappropriate_content(content):
            return "The analysis detected potentially inappropriate content in the response. Please review the audio content for appropriateness. Score: 0/100"
            
        return content
    except Exception as e:
        return f"Error analyzing answer: {str(e)}. Score: 0/100"

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
    temp_input = None
    temp_wav = None
    try:
        # Check if the uploaded_file is a string (file path) or a file object
        if isinstance(uploaded_file, str):
            # It's a file path
            file_path = uploaded_file
            file_extension = file_path.split('.')[-1].lower()
            temp_input = file_path
        else:
            # It's a file object (from streamlit uploader)
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Create a temporary file for the uploaded audio
            temp_dir = tempfile.gettempdir()
            temp_input = os.path.join(temp_dir, f"input_audio.{file_extension}")
            
            # Save uploaded file
            with open(temp_input, 'wb') as f:
                f.write(uploaded_file.getvalue())

        if file_extension not in SUPPORTED_FORMATS:
            st.error(f"Unsupported file format. Please upload one of: {', '.join(SUPPORTED_FORMATS.keys())}")
            return None, None

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
        avg_pitch = np.mean([s['pitch'] for s in segments]) if segments else 0

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
        # Clean up temporary files - only if they're not the original input file
        if isinstance(uploaded_file, str):
            # If it's a file path, only clean temp_wav if it's different from the input
            if temp_wav and temp_wav != temp_input and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
        else:
            # If it's a file object, clean both temp files
            for temp_file in [temp_input, temp_wav]:
                if temp_file and os.path.exists(temp_file):
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

    prompt = f"""Analyze the provided speech transcription and voice metrics. Deliver concise, structured feedback addressing:

1. **Voice Quality:** Comment briefly on overall clarity and consistency of voice.
2. **Pitch Modulation:** Evaluate the appropriateness and variation of pitch.
3. **Pacing & Pauses:** Assess the number and duration of pauses—highlight if pacing enhances or disrupts delivery.
4. **Content Clarity:** Quickly note if content structure and message clarity were effective.
5. **Improvement Suggestions:** Provide clear, actionable recommendations focusing specifically on pitch and pause management.

Speech Analysis Data:
- Transcription: {transcription}
- Average Pitch: {speech_metrics['average_pitch']:.2f} Hz
- Total Pauses: {speech_metrics['pause_count']}
- Average Pause Duration: {speech_metrics['average_pause_duration']} seconds
- Pitch Variation: from {min(pitch_variations):.1f}% to {max(pitch_variations):.1f}% compared to the average.

Format your response clearly, succinctly, and in easily readable bullet points. Keep feedback professional, specific, and positive in tone.
"""

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
        .warning-container {
            background-color: #8B0000;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🎤 AI Interview Preparation Platform</h1>', unsafe_allow_html=True)

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Resume Analysis & Questions", "Speech Analysis"])

    with tab1:
        st.subheader("Resume Analysis & Question Generation")
        
        # Resume upload
        st.markdown("### Upload Your Resume")
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")
        
        # Job description section
        st.markdown("### Job Description (Optional)")
        job_description_method = st.radio(
            "Choose how to provide job description:",
            ["None", "Upload PDF", "Type job description"]
        )
        
        job_description = None
        
        if job_description_method == "Upload PDF":
            uploaded_job_desc = st.file_uploader("Upload Job Description (PDF)", type="pdf", key="job_desc_pdf")
            if uploaded_job_desc:
                job_description_text, error = extract_text_from_pdf(uploaded_job_desc)
                if error:
                    st.error(error)
                else:
                    job_description = job_description_text
                    with st.expander("View Extracted Job Description"):
                        st.text(job_description)
        
        elif job_description_method == "Type job description":
            job_description_input = st.text_area("Enter job description", height=200)
            if job_description_input.strip():
                # Check for inappropriate content
                if contains_inappropriate_content(job_description_input):
                    st.markdown('<div class="warning-container">The job description contains potentially inappropriate content. Please provide appropriate content.</div>', unsafe_allow_html=True)
                else:
                    job_description = job_description_input
        
        # Question generation
        num_questions = st.slider("Select number of questions", 3, 10, 5)
        
        if uploaded_resume:
            resume_text, error = extract_text_from_pdf(uploaded_resume)
            
            if error:
                st.error(error)
            else:
                st.session_state.resume_text = resume_text
                
                with st.expander("View Extracted Resume Text"):
                    st.text(resume_text)
                
                # Store job description in session state
                if job_description:
                    st.session_state.job_description = job_description
                elif 'job_description' in st.session_state:
                    del st.session_state.job_description
                
                if st.button("Generate Interview Questions"):
                    with st.spinner("Generating questions..."):
                        questions, error = generate_questions(
                            resume_text, 
                            num_questions, 
                            job_description=job_description if job_description else None
                        )
                        
                        if error:
                            st.error(error)
                        else:
                            # Validate we have the correct number of questions
                            if len(questions) != num_questions:
                                st.warning(f"Generated {len(questions)} valid questions instead of the requested {num_questions}. You may want to regenerate.")
                            
                            if len(questions) > 0:
                                st.session_state.questions = questions
                                st.success("Questions generated successfully!")
                                
                                # Display information about question generation
                                if job_description:
                                    st.info("Questions are tailored based on both your resume and the job description.")
                                else:
                                    st.info("Questions are based solely on your resume. Add a job description for more targeted questions.")
                            else:
                                st.error("Failed to generate valid questions. Please check your resume content and try again.")

        # Question and Answer Section
        if "questions" in st.session_state and st.session_state.questions:
            st.subheader("Practice Your Answers")
            audio_files = {}

            for i, question in enumerate(st.session_state.questions, 1):
                st.markdown(f"**Question {i}:** {question}")
                
                uploaded_audio = st.file_uploader(
                    f"Upload your answer for Question {i}",
                    type=list(SUPPORTED_FORMATS.keys()),
                    key=f"audio_{i}"
                )

                if uploaded_audio:
                    # Check file size to ensure it's not empty
                    if uploaded_audio.size < 1000:  # Less than 1KB
                        st.error(f"The uploaded audio file for Question {i} appears to be empty or corrupted. Please upload a valid audio file.")
                        continue
                        
                    temp_audio_path = os.path.join(tempfile.gettempdir(), f"answer_{i}.wav")
                    
                    try:
                        # Convert to WAV if needed
                        if uploaded_audio.name.split('.')[-1].lower() != 'wav':
                            with open(os.path.join(tempfile.gettempdir(), f"temp_input_{i}.{uploaded_audio.name.split('.')[-1]}"), 'wb') as f:
                                f.write(uploaded_audio.getvalue())
                                
                            audio = AudioSegment.from_file(f.name, format=uploaded_audio.name.split('.')[-1].lower())
                            audio.export(temp_audio_path, format="wav")
                        else:
                            with open(temp_audio_path, "wb") as f:
                                f.write(uploaded_audio.getvalue())
                        
                        audio_files[f"audio_{i}"] = temp_audio_path
                        st.success(f"Answer {i} uploaded successfully!")
                    except Exception as e:
                        st.error(f"Error processing audio file for Question {i}: {str(e)}")

            st.session_state.audio_files = audio_files

            if "audio_files" in st.session_state and st.session_state.audio_files and st.button("Analyze All Answers"):
                st.subheader("Comprehensive Answer Analysis")
                
                job_desc_for_analysis = st.session_state.get('job_description', None)
                resume_text_for_analysis = st.session_state.get('resume_text', None)
                
                all_scores = []
                
                for i, question in enumerate(st.session_state.questions, 1):
                    audio_key = f"audio_{i}"
                    if audio_key in st.session_state.audio_files:
                        st.markdown(f"### Analysis for Question {i}")
                        
                        # Content Analysis
                        with st.spinner(f"Analyzing content for Question {i}..."):
                            content_analysis = analyze_answer_content(
                                st.session_state.audio_files[audio_key],
                                question,
                                job_description=job_desc_for_analysis,
                                resume_text=resume_text_for_analysis
                            )
                            st.markdown("#### Content Analysis")
                            st.write(content_analysis)
                            
                            # Extract score if available
                            score_match = re.search(r'(\d+)[\/\s]*100', content_analysis)
                            if score_match:
                                all_scores.append(int(score_match.group(1)))
                        
                        # Speech Analysis
                        with st.spinner(f"Analyzing speech patterns for Question {i}..."):
                            try:
                                transcription, speech_metrics = analyze_audio(
                                    st.session_state.audio_files[audio_key]
                                )
                                
                                if transcription and speech_metrics:
                                    st.markdown("#### Speech Analysis")
                                    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                                    
                                    # Display Metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("🎵 Average Pitch", f"{speech_metrics['average_pitch']:.2f} Hz")
                                    with col2:
                                        st.metric("⏱️ Pauses", f"{speech_metrics['pause_count']}")
                                    with col3:
                                        st.metric("📝 Word Count", str(len(transcription.split())))
                                    
                                    # Display Pitch Graph
                                    st.subheader("Pitch Variation")
                                    pitch_data = [segment['pitch'] for segment in speech_metrics['segments']]
                                    st.line_chart(pitch_data)
                                    
                                    # Generate Speech Feedback
                                    feedback = generate_feedback(transcription, speech_metrics)
                                    st.markdown("#### Speech Feedback")
                                    st.write(feedback)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.error("Could not extract speech data from the audio file. The file may be empty or in an unsupported format.")
                            except Exception as e:
                                st.error(f"Error analyzing speech patterns: {str(e)}")
                
                # Display overall score if we have scores
                if all_scores:
                    avg_score = sum(all_scores) / len(all_scores)
                    st.markdown("### Overall Assessment")
                    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                    st.metric("📊 Overall Job Fit Score", f"{avg_score:.1f}/100")
                    
                    # Provide overall feedback based on score
                    if avg_score >= 80:
                        st.success("Excellent performance! You appear to be a strong fit for this position.")
                    elif avg_score >= 60:
                        st.info("Good performance. With some improvements in the areas mentioned, you could be a solid candidate.")
                    elif avg_score >= 40:
                        st.warning("Moderate performance. Consider working on the suggestions provided to improve your candidacy.")
                    else:
                        st.error("You may need significant preparation before applying for this position. Focus on the key areas mentioned in the feedback.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Speech Analysis")
        # Original speech analysis functionality
        uploaded_file = st.file_uploader(
            f"Upload an audio file for speech analysis (Supported formats: {', '.join(SUPPORTED_FORMATS.keys())})", 
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
