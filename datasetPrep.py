import os
import csv
from scripts.utils import emotion_label_tagger, emotion_label_tagger_crema
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict
import numpy as np

data_path = "data/raw"

base_dirs = {
    0: "RAVDESS",
    1: "ESD",
    2: "CREMA-D",
}

valid_emotions = {
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprise",
}
exclude_emotion = {"disgust", "fearful", "surprise"}
metadata = []

rasd_path = os.path.join(data_path, base_dirs[0])

for root, dirs, files in os.walk(rasd_path):
    for file in files:
        if file.endswith(".wav"):

            emotion = file.split("-")[2]
            emotion = emotion_label_tagger(emotion)
            if emotion == "calm":
                emotion = "neutral"

            if emotion in exclude_emotion:
                continue
            audio_path = os.path.join(root, file)
            audio_name = audio_path.split("\\")[-1]
            metadata.append(
                {
                    "path": audio_path,
                    "name": audio_name,
                    "emotion": emotion,
                }
            )


for root, dirs, files in os.walk(os.path.join(data_path, base_dirs[1])):
    for file in files:
        audio_path = os.path.join(root, file)
        emotion_label = os.path.basename(os.path.dirname(audio_path)).lower()

        if emotion_label not in valid_emotions:
            continue  # skip unknown or misnamed labels
        if emotion_label == "calm":
            emotion_label = "neutral"
        if emotion_label in exclude_emotion:
            continue
        audio_name = os.path.basename(audio_path)
        metadata.append(
            {
                "path": audio_path,
                "name": audio_name,
                "emotion": emotion_label,
            }
        )

for root, dirs, files in os.walk(os.path.join(data_path, base_dirs[2], "AudioWAV")):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            audio_name = audio_path.split("\\")[-1]
            emotion_id = audio_name.split("_")[2]
            emotion = emotion_label_tagger_crema(emotion_id)

            if emotion in exclude_emotion:
                continue
            metadata.append(
                {
                    "path": audio_path,
                    "name": audio_name,
                    "emotion": emotion,
                }
            )


# Count samples per class
# 1. Count samples per class in the original metadata
# class_counts = defaultdict(int)
# for item in metadata:
#     class_counts[item["emotion"]] += 1

# # 2. Calculate duplication factors
# target_count = max(class_counts.values())
# duplication_factors = {
#     cls: int(target_count / count) for cls, count in class_counts.items()
# }

# # 3. Duplicate paths for underrepresented classes
# balanced_data = []
# for item in metadata:
#     cls = item["emotion"]
#     # Add the original item and duplicates
#     for _ in range(duplication_factors[cls]):
#         balanced_data.append(item.copy())  # Shallow copy of dict

# # 4. Count classes in the balanced set and print
balanced_class = defaultdict(int)
for item in metadata:
    balanced_class[item["emotion"]] += 1


print("Balanced class counts:")
for emotion, count in balanced_class.items():
    print(f"{emotion}: {count}")

train_metadata, test_metadata = train_test_split(
    metadata,
    test_size=0.2,
    stratify=[item["emotion"] for item in metadata],
    shuffle=True,
    random_state=42,
)

with open("data/train_metadata.json", "w") as f:
    json.dump(train_metadata, f)

with open("data/test_metadata.json", "w") as f:
    json.dump(test_metadata, f)

print("Train & Validation metadatas are Saved and Ready")
