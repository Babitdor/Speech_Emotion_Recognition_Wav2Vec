import torch
import librosa
import streamlit as st
import tempfile
import os
import numpy as np
from scripts.pretrained_model import Wav2Vec2ForAudioClassification
from transformers import Wav2Vec2Processor
import sys
from typing import List, Tuple

# Windows asyncio fix for Streamlit + librosa
if sys.platform.startswith("win"):
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

MODEL_PATH = "models/Output"


@st.cache_resource
def load_model_and_processor() -> (
    Tuple[Wav2Vec2Processor, torch.nn.Module, str, List[str]]
):
    """Load and cache the model, processor, and labels."""
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
        model = Wav2Vec2ForAudioClassification.from_pretrained(MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Get labels in correct order
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        return processor, model, device, labels
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise


def preprocess_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and preprocess audio file."""
    try:
        waveform, sr = librosa.load(audio_path, sr=target_sr)
        # Normalize audio
        waveform = librosa.util.normalize(waveform)
        return waveform
    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        raise


def predict_emotion(audio_path: str) -> List[Tuple[str, float]]:
    """Predict emotion probabilities from audio file."""
    try:
        # Preprocess audio
        waveform = preprocess_audio(audio_path)

        # Process with feature extractor
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_attention_mask=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Model inference
        with torch.no_grad():
            outputs = model(inputs["input_values"])
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return list(zip(LABELS, probs))
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        raise


def display_results(emotion_scores: List[Tuple[str, float]]):
    """Display prediction results in Streamlit."""
    st.subheader("Emotion Probabilities")

    # Sort emotions by probability (descending)
    sorted_scores = sorted(emotion_scores, key=lambda x: x[1], reverse=True)

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Display all probabilities in a clean format
        for emotion, score in sorted_scores:
            st.metric(
                label=emotion.capitalize(),
                value=f"{score*100:.1f}%",
                help=f"Confidence: {score*100:.2f}%",
            )

    with col2:
        # Visualize top emotion
        top_emotion, top_score = sorted_scores[0]
        st.success(
            f"**Predicted Emotion**: {top_emotion.capitalize()}\n\n"
            f"**Confidence**: {top_score*100:.1f}%"
        )

        # Add a simple bar chart visualization
        st.bar_chart({emotion: score for emotion, score in sorted_scores}, height=300)


# Load model and processor
try:
    processor, model, device, LABELS = load_model_and_processor()
except:
    st.error("Failed to initialize model. Please check the model files.")
    st.stop()

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a WAV audio file to analyze its emotional content.")

uploaded_file = st.file_uploader(
    "Choose a WAV file", type=["wav"], help="Supported formats: WAV (16kHz recommended)"
)

if uploaded_file is not None:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Display audio player
    st.audio(temp_path, format="audio/wav")

    # Process and display results
    with st.spinner("Analyzing emotion..."):
        try:
            emotion_scores = predict_emotion(temp_path)
            display_results(emotion_scores)
        except:
            st.error("Error during emotion analysis")
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
