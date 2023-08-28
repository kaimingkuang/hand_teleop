import torch.nn as nn


class DomainClfLoss(nn.Module):

    def __init__(self, w_domain):
        super().__init__()

        self.w_domain = w_domain
        self.action_loss_fn = nn.MSELoss()
        self.domain_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, action_output, action_target, domain_output,
            domain_target):
        action_loss = self.action_loss_fn(action_output, action_target)
        domain_loss = self.domain_loss_fn(domain_output, domain_target)
        total_loss = action_loss + self.w_domain * domain_loss

        return total_loss, action_loss, domain_loss
