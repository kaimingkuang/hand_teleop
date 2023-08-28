import torch
from torch.autograd import Function
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class ClassToken(nn.Module):

    def __init__(self, cls_token_idx=0):
        super().__init__()
        self.cls_token_idx = cls_token_idx

    def forward(self, tokens):
        if self.cls_token_idx == 0:
            img_feats = tokens["img_feats"][:, self.cls_token_idx]
            return img_feats
        else:
            return tokens["img_feats"].squeeze(dim=-1).squeeze(dim=-1)


class MLPLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )


class MLP(nn.Sequential):

    def __init__(self, in_channels, hidden_channels, out_channels=None):
        n_channels = [in_channels] + hidden_channels
        layers = []
        for in_ch, out_ch in zip(n_channels[:-1], n_channels[1:]):
            layers.append(MLPLayer(in_ch, out_ch))

        if out_channels is not None:
            layers.append(nn.Linear(hidden_channels[-1], out_channels))
        super().__init__(*layers)


class PolicyNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.backbone = MLP(in_channels, hidden_channels)
        self.out_layer = nn.Linear(hidden_channels[-1], out_channels)
    
    def forward(self, inputs, ret_feats=False):
        feats = self.backbone(inputs)
        outputs = self.out_layer(feats)

        if ret_feats:
            return outputs, feats
        else:
            return outputs


class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


revgrad = GradientReversalFunc.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


class Agent(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.vision_net = self.init_vision_net()

        self.policy_net = self.init_policy_net()

        if args.grad_rev:
            feat_channels = args.hidden_channels[-1]
            self.domain_clf = nn.Sequential(
                GradientReversal(),
                MLP(feat_channels, [feat_channels // 2], 1),
            )

    def init_vision_net(self):
        model_fn = eval(f"models.{self.args.backbone}")
        model = model_fn(weights="IMAGENET1K_V1")
        vision_net = nn.Sequential(
            create_feature_extractor(model, {"avgpool": "img_feats"}),
            ClassToken("img_feats")
        )

        return vision_net

    def init_policy_net(self):
        model = PolicyNet(
            self.args.in_channels,
            self.args.hidden_channels,
            self.args.out_channels
        )

        return model

    def forward(self, robot_qpos, vision_inputs=None, use_vision_net=False):
        if use_vision_net:
            n_steps = vision_inputs.size(1)
            vision_feats = []
            for i in range(n_steps):
                vision_feats.append(self.vision_net(vision_inputs[:, i]))
            vision_feats = torch.cat(vision_feats, dim=0)
            inputs = torch.cat([vision_feats, robot_qpos], dim=-1)
            inputs = inputs.reshape((1, -1))
        else:
            inputs = torch.cat([vision_inputs, robot_qpos], dim=-1)

        if self.args.grad_rev:
            outputs, feats = self.policy_net(inputs, ret_feats=True)
            domain_outputs = self.domain_clf(feats).squeeze(dim=-1)
            return outputs, domain_outputs
        else:
            outputs = self.policy_net(inputs)
            return outputs

    @torch.no_grad()
    def get_action(self, robot_qpos, vision_inputs):
        if self.args.grad_rev:
            action, _ = self(robot_qpos, vision_inputs, use_vision_net=True)
        else:
            action = self(robot_qpos, vision_inputs, use_vision_net=True)
        action = action.cpu().numpy()

        return action


if __name__ == "__main__":
    import torch


    args = {
        "backbone": "regnet_y_1_6gf",
        "use_vision_net": True,
        "in_channels": 3668,
        "hidden_channels": [1200, 600, 300],
        "out_channels": 22
    }
    model = Agent(args)
    vision_inputs = torch.rand(4, 4, 3, 224, 224)
    robot_qpos = torch.rand(4, 116)
    outputs = model(robot_qpos, vision_inputs)
    print(1)
