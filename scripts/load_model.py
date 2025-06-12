from transformers import Wav2Vec2Processor, AutoConfig
from scripts.pretrained_model import Wav2Vec2ForAudioClassification
import torch


def load_model_and_processor(model_name, label_list, class_weights, mode):

    config = AutoConfig.from_pretrained(
        model_name,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )

    setattr(config, "pooling_mode", mode)

    if not hasattr(config, "classifier_dropout"):
        config.classifier_dropout = 0.1

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForAudioClassification.from_pretrained(
        model_name,
        config=config,
    )

    model.class_weights = class_weights

    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    return model, processor
