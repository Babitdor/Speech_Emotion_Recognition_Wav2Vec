from transformers import Trainer
import torch.nn.functional as F
from scripts.loss import CenterLoss, focal_loss


class CTCTrainer(Trainer):
    def __init__(
        self,
        *args,
        num_classes,
        class_weights=None,
        center_loss_weight=0.1,
        center_loss_feat_dim,
        focal_loss_weight=0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.center_loss_weight = center_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.center_loss_fn = CenterLoss(
            num_classes=num_classes,
            feat_dim=center_loss_feat_dim,
            device=self.args.device,
            lambda_c=center_loss_weight,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits

        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=(
                self.class_weights.to(logits.device)
                if self.class_weights is not None
                else None
            ),
        )

        fl_loss = focal_loss(logits=logits, targets=labels)

        features = None
        if hasattr(outputs, "hidden_states"):
            features = outputs.hidden_states[-1][:, 0, :]
        elif hasattr(outputs, "features"):
            features = outputs.features
        else:
            features = None

        cl_loss = 0.0
        if self.model.training and features is not None:
            cl_loss = self.center_loss_fn(features, labels)
            cl_loss = cl_loss / features.size(1)

        loss = (
            ce_loss
            + (self.focal_loss_weight * fl_loss if self.model.training else 0.0)
            + (cl_loss if self.model.training else 0.0)
        )

        return (loss, outputs) if return_outputs else loss
