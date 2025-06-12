import os
import re


def emotion_label_tagger(emotion_code):
    labels = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprise",
    }
    return labels.get(emotion_code, "unknown")


def emotion_label_tagger_crema(emotion_code):
    labels = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fearful",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }
    return labels.get(emotion_code, "unknown")


def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # Sort by checkpoint number
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r"\d+", x)[0]))
    return os.path.join(output_dir, checkpoints[-1])
