# Speech Emotion Recognition with Wav2Vec2

This project implements **Speech Emotion Recognition (SER)** using [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) and the Hugging Face Transformers library. It supports training, evaluation, and inference on custom or public emotion-labeled speech datasets.

---

## Features

- **Wav2Vec2-based** audio feature extraction and classification
- Customizable data augmentation and class balancing
- Handles class imbalance with weighted loss
- Training, validation, and inference scripts
- Streamlit web app for interactive audio emotion prediction (supports file upload and voice recording)
- TensorBoard logging and checkpointing

---

## Project Structure

```
.
├── data/                   # Place your train/test metadata and audio files here
├── scripts/
│   ├── dataloader.py       # SERDataLoader: dataset class
│   ├── datacollator.py     # Data collator for batching
│   ├── loss.py             # Custom loss functions (center, focal)
│   ├── pretrained_model.py # Wav2Vec2ForAudioClassification model
│   ├── augmentation.py     # Audio augmentation utilities
│   ├── eval.py             # compute_metrics for evaluation
│   └── utils.py            # Utility functions
├── train.py                # Main training script
├── inference.py            # Streamlit inference app
├── datasetPrep.py          # Data preparation and balancing
└── requirements.txt        # Python dependencies
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- Place your audio files and metadata JSONs in the `data/` directory.
- Use `datasetPrep.py` to balance and split your dataset.

### 3. Train the Model

```bash
python train.py
```

- Training progress and logs will be saved in the `models/` and `logs/` directories.

### 4. Run Inference (Web App)

```bash
streamlit run inference.py
```

- Upload a `.wav` file or record your voice to get emotion predictions.

---

## Configuration

- **Model:** `facebook/wav2vec2-base-960h` (can be changed in `train.py`)
- **Pooling:** Mean pooling by default
- **Class weights:** Computed automatically for imbalanced datasets
- **Augmentation:** Controlled in `scripts/augmentation.py`
- **Metrics:** F1, accuracy, precision, recall

---

## Example Usage

**Training:**

```bash
python train.py
```

**Inference:**

```bash
streamlit run inference.py
```

---

## Customization

- **Add new emotions:** Update your metadata and label mapping in `train.py`.
- **Change model:** Modify `MODEL_NAME` in `train.py`.
- **Tune hyperparameters:** Edit `TrainingArguments` in `train.py`.
- **Augmentation:** Edit `scripts/augmentation.py` for custom audio augmentations.

---

## Citation

If you use this project, please cite the original [Wav2Vec2 paper](https://arxiv.org/abs/2006.11477) and Hugging Face Transformers.

---

## License

This project is licensed under the MIT License.

---
