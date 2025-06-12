import json
from scripts.dataloader import SERDataLoader
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from scripts.load_model import load_model_and_processor
from scripts.datacollator import DataCollatorCTCWithPadding
from scripts.augmentation import apply_augmentation
from transformers import TrainingArguments, Trainer
import numpy as np

# from scripts.trainer import CTCTrainer
from scripts.eval import compute_metrics
from scripts.utils import get_last_checkpoint
import torch
import os

MODEL_NAME = "facebook/wav2vec2-base-960h"
POOLING_MODE = "mean"

if torch.cuda.is_available():
    print("Training using CUDA")
else:
    print("No CUDA available ---> Training using CPU instead")


train_metadata_path = r"data\train_metadata.json"
val_metadata_path = r"data\test_metadata.json"

with open(train_metadata_path, "r") as f:
    train_metadata = json.load(f)
with open(val_metadata_path, "r") as f:
    val_metadata = json.load(f)

# Combine and get unique emotions
all_metadata = train_metadata + val_metadata
label_list = sorted(set(item["emotion"] for item in all_metadata))
label2id = {label: idx for idx, label in enumerate(label_list)}

# Print emotion counts
emotion_counts = Counter(item["emotion"] for item in all_metadata)
print("Emotion counts:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")

# Convert emotions to label indices for training
y_train = [label2id[item["emotion"]] for item in train_metadata]

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(label_list)),
    y=y_train,
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
# Loading Model and Processor
model, processor = load_model_and_processor(
    model_name=MODEL_NAME,
    label_list=label_list,
    class_weights=class_weights,
    mode=POOLING_MODE,
)

collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding=True,
)

training_args = TrainingArguments(
    output_dir="models",
    eval_strategy="steps",  # Evaluate at the end of each epoch
    save_strategy="steps",  # Save once per epoch
    logging_strategy="steps",
    eval_steps=500,
    log_level="error",
    save_steps=500,
    logging_dir="logs",
    max_grad_norm=1.0,
    report_to="tensorboard",  # Log to TensorBoard
    logging_steps=50,  # More frequent logging for insight
    per_device_train_batch_size=8,  # Increase if GPU memory allows
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Simulates larger effective batch size
    num_train_epochs=20,  # More epochs for fine-tuning speech models
    learning_rate=3e-5,  # Lower LR for stable fine-tuning
    warmup_steps=200,  # Warmup helps with convergence
    weight_decay=0.01,  # Regularization
    fp16=True,  # Good if using a modern GPU
    save_total_limit=3,
    metric_for_best_model="eval_f1",
    load_best_model_at_end=True,  # Automatically keep best checkpoint
    greater_is_better=True,
)


train_dataset = SERDataLoader(
    metadata_file=train_metadata_path,
    processor=processor,
    transform=True,
    label_map=label2id,
)

val_dataset = SERDataLoader(
    metadata_file=val_metadata_path, processor=processor, label_map=label2id
)

print("Training Dataset Size: ", train_dataset.__len__())
print("Validation Dataset Size: ", val_dataset.__len__())

trainer = Trainer(
    model=model,
    data_collator=collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

checkpoint = get_last_checkpoint(training_args.output_dir)
trainer.train(resume_from_checkpoint=checkpoint)

model.save_pretrained(os.path.join(training_args.output_dir, "Output"))
processor.save_pretrained(os.path.join(training_args.output_dir, "Output"))
