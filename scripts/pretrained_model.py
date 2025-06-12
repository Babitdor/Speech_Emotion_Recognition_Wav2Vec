import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from scripts.loss import CenterLoss, focal_loss
from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)


@dataclass
class SpeechClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Optimized classification head with sequential operations."""

    def __init__(self, config):
        super().__init__()
        layers = [
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels),
        ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, features):
        return self.classifier(features)


class Wav2Vec2ForAudioClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode

        # Model components
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        # Initialize weights
        self.init_weights()

        # Loss configuration with defaults
        self.center_loss_weight = getattr(config, "center_loss_weight", 0.1)
        self.focal_loss_weight = getattr(config, "focal_loss_weight", 0.5)
        self.class_weights = None  # Set externally if needed

        # Initialize center loss
        self._init_center_loss()

    def _init_center_loss(self):
        """Initialize center loss with proper device handling."""
        self.center_loss_fn = CenterLoss(
            num_classes=self.config.num_labels,
            feat_dim=self.config.hidden_size,
            device=self.device,
            lambda_c=self.center_loss_weight,
        )

    @property
    def device(self):
        """Helper property to get the device."""
        return next(self.parameters()).device

    def freeze_feature_extractor(self):
        """Freeze wav2vec2 feature extractor parameters."""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        """Optimized pooling operations."""
        if mode == "mean":
            return hidden_states.mean(dim=1)
        elif mode == "sum":
            return hidden_states.sum(dim=1)
        elif mode == "max":
            return hidden_states.amax(dim=1)
        else:
            raise ValueError(
                f"Invalid pooling mode: {mode}. Must be one of ['mean', 'sum', 'max']"
            )

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Forward pass through wav2vec2
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Pooling and classification
        hidden_states = outputs[0]
        pooled = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(pooled)

        # Calculate loss if labels provided
        loss = (
            self._compute_loss(logits, pooled, labels) if labels is not None else None
        )

        if not return_dict:
            return (loss, logits, outputs.hidden_states, outputs.attentions)

        return SpeechClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_loss(self, logits, hidden_states, labels):
        """Compute combined loss (CE + Focal + Center)."""
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=(
                self.class_weights.to(self.device)
                if self.class_weights is not None
                else None
            ),
        )

        # Focal Loss
        fl_loss = focal_loss(logits=logits, targets=labels) * self.focal_loss_weight

        # Center Loss (normalized by feature dimension)
        cl_loss = self.center_loss_fn(hidden_states, labels) / hidden_states.size(1)

        return ce_loss + fl_loss + cl_loss
