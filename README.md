# Speech Emotion Recognition using Wav2Vec 2.0
This repository implements a Speech Emotion Recognition (SER) system by fine-tuning a pretrained Wav2Vec 2.0 model on a combined dataset of RAVDESS and ESD (Emotional Speech Database). The goal is to classify emotions from spoken audio using a deep learning approach.

This project leverages the Wav2Vec 2.0 model, a powerful pretrained speech-to-text model developed by Facebook AI, and fine-tunes it for the task of speech emotion recognition. The fine-tuning is performed using a combined dataset of the RAVDESS and ESD datasets, which contain audio clips with labeled emotional expressions.

## Key features:

Pretrained Wav2Vec 2.0 model for feature extraction
Fine-tuning on emotion-labeled speech datasets
Evaluation of emotion classification performance

## Datasets
The project combines two publicly available emotion-labeled speech datasets:

### RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song)

A dataset consisting of 24 actors (12 male and 12 female), with each recording 8 emotions (calm, happy, sad, angry, fearful, surprise, disgust, and neutral) in both speech and song.
Contains 2,880 clips in total.

### ESD (Emotional Speech Database)

A dataset containing emotionally varied speech in multiple languages.
Includes 3,000+ recordings across a wide range of emotional categories.
The data is preprocessed, including resampling, trimming, and converting audio to spectrograms or mel-frequency cepstral coefficients (MFCCs) for training the model.

## Model
The base model used is Wav2Vec 2.0, a self-supervised model for learning representations of raw speech audio. It has been shown to perform well for speech-related tasks, including Automatic Speech Recognition (ASR). For this project, the model is fine-tuned to predict emotions from speech instead of transcribing it into text.

## Architecture:
Wav2Vec 2.0 for extracting features from raw audio input.
Fully Connected Layers (FC) for emotion classification after extracting features.
Softmax layer for multi-class emotion classification (happy, sad, angry, etc.).

## Installation
Follow these steps to set up the environment for this project:

1. Clone the repository:

     ```bash
     git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
     cd Speech-Emotion-Recognition

2. Create a virtual environment

     ```bash 
     python -m venv .env
     run : .env/Scripts/Activate

3. Install the required dependencies:

     ```bash
     pip install -r requirements.txt
     
# Evaluation
### Metrics: 
     Accuracy, Precision, Recall, F1-score
### Loss function: 
     Cross-entropy loss
### Optimizer: 
     Adam optimizer with learning rate decay

After training, the model is evaluated on a separate test set using common classification metrics.


# NOTE: 
The model in it's current state needs to be trained on more data. 
Currently the model has been trained on a subset size of 5000, because of hardware limitations.
But with more data, it is capable of performing well than the current state.
