import torch
import torch.nn.functional as F


class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, device="cpu", lambda_c=1.0):
        super().__init__()
        self.centers = torch.nn.Parameter(
            torch.randn(num_classes, feat_dim, device=device)
        )
        self.lambda_c = lambda_c

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        # Use .pow(2).sum(dim=1).mean() for efficiency
        center_loss = ((features - centers_batch).pow(2).sum(dim=1)).mean() * 0.5
        return self.lambda_c * center_loss


def focal_loss(logits, targets, alpha=1, gamma=2, reduction="mean"):
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean() if reduction == "mean" else focal
