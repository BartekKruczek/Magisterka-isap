import torch
import torch.nn as nn

from metrics import CustomMetrics

class TEDLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, alpha: float = 0.1):
        super(TEDLoss, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.metrics = CustomMetrics()

    def forward(self, outputs, targets, generated_json, ground_truth_json):
        loss = self.base_loss(outputs, targets)

        ted = self.metrics.calculate_tree_edit_distance(generated_json, ground_truth_json)
        ted_scaled = self.alpha * torch.tensor(ted, dtype=torch.float32)

        return loss + ted_scaled