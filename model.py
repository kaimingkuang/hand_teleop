import torch
import torch.nn.functional as F
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
            nn.BatchNorm1d(out_channels),
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

    def __init__(
            self,
            vis_dims,
            qpos_dims,
            hidden_channels,
            out_channels,
            n_vis_layers,
            n_policy_layers,
            drop_prob,
        ):
        super().__init__()

        self.vis_lins = nn.ModuleList(
            [nn.Linear(vis_dims, hidden_channels)]\
            + [nn.Linear(hidden_channels, hidden_channels) for _ in range(n_vis_layers - 1)]
        )
        self.vis_sim_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels)
            for _ in range(n_vis_layers)])
        self.vis_real_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels)
            for _ in range(n_vis_layers)])

        self.policy_lins = nn.ModuleList(
            [nn.Linear(hidden_channels + qpos_dims, hidden_channels)]\
            + [nn.Linear(hidden_channels, hidden_channels) for _ in range(n_policy_layers - 1)],
        )
        self.policy_sim_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels)
            for _ in range(n_policy_layers)])
        self.policy_real_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels)
            for _ in range(n_policy_layers)])

        self.out_layer = nn.Linear(hidden_channels, out_channels)
        self.drop_prob = drop_prob

    def forward(self, vis_feats, qpos, mode):
        if mode == "sim":
            for i in range(len(self.vis_lins)):
                vis_feats = F.relu(self.vis_sim_norms[i](self.vis_lins[i](vis_feats)))
        else:
            for i in range(len(self.vis_lins)):
                vis_feats = F.relu(self.vis_real_norms[i](self.vis_lins[i](vis_feats)))

        vis_feats = F.dropout(vis_feats, self.drop_prob, self.training)
        feats = torch.cat([vis_feats, qpos], dim=-1)

        if mode == "sim":
            for i in range(len(self.policy_lins)):
                feats = F.relu(self.policy_sim_norms[i](self.policy_lins[i](feats)))
        else:
            for i in range(len(self.policy_lins)):
                feats = F.relu(self.policy_real_norms[i](self.policy_lins[i](feats)))

        outputs = self.out_layer(F.dropout(feats, self.drop_prob, self.training))

        return outputs


class Agent(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.vision_net = self.init_vision_net()

        self.policy_net = self.init_policy_net()

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
            self.args.vis_dims,
            self.args.qpos_dims,
            self.args.hidden_channels,
            self.args.out_channels,
            self.args.n_vis_layers,
            self.args.n_policy_layers,
            self.args.drop_prob,
        )

        return model

    def forward(self, images, robot_qpos, mode):
        vis_feats = self.get_image_feats(images)
        vis_feats = vis_feats.unfold(0, self.args.window_size, 1)\
            .permute((0, 2, 1))

        b = vis_feats.size(0)
        vis_feats = vis_feats.reshape((b, -1))
        robot_qpos = robot_qpos.unfold(0, self.args.window_size, 1)\
            .permute((0, 2, 1)).reshape((b, -1))

        outputs = self.policy_net(vis_feats, robot_qpos, mode)

        return outputs

    @torch.no_grad()
    def get_action(self, images, qpos, mode):
        action = self(images, qpos, mode)
        action = action.cpu().numpy()

        return action

    def get_image_feats(self, images):
        if self.vision_net.training:
            feats = self.vision_net(images)
        else:
            with torch.no_grad():
                feats = self.vision_net(images)

        return feats


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
