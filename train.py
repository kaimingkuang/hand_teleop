import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from trainer import Trainer


def parse_args():
    parser = ArgumentParser()

    # required
    parser.add_argument("--demo-folder", required=True)
    parser.add_argument("--seed", required=True, type=int)

    # data
    parser.add_argument("--robot", default="xarm6_allegro_modified_finger")

    # model
    parser.add_argument("--backbone", default="regnet_y_3_2gf")
    parser.add_argument("--vis-dims", default=6048, type=int)
    parser.add_argument("--qpos-dims", default=116, type=int)
    parser.add_argument("--hidden-channels", default=1024, type=int)
    parser.add_argument("--n-vis-layers", default=2, type=int)
    parser.add_argument("--n-policy-layers", default=3, type=int)
    parser.add_argument("--drop-prob", default=0.2, type=float)
    parser.add_argument("--out-channels", default=22, type=int)
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--window-size", default=4, type=int)

    # training
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd-coef", default=0.01, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--grad-acc", default=1, type=int)
    parser.add_argument("--aug-prob", default=0.5, type=float)

    # evaluation
    parser.add_argument("--eval-x-steps", default=4, type=int)
    parser.add_argument("--eval-y-steps", default=5, type=int)
    parser.add_argument("--final-x-steps", default=10, type=int)
    parser.add_argument("--final-y-steps", default=10, type=int)
    parser.add_argument("--val-pct", default=0.1, type=float)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-beg", default=200, type=int)
    parser.add_argument("--eval-freq", default=100, type=int)
    parser.add_argument("--max-eval-steps", default=2000, type=int)

    # others
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n-workers", default=8, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--wandb-off", action="store_true")

    args = parser.parse_args()

    if os.path.basename(args.demo_folder).startswith("pick_place"):
        args.task = "pick_place"
    elif os.path.basename(args.demo_folder).startswith("dclaw"):
        args.task = "dclaw"

    if args.debug:
        args.n_workers = 0
        args.eval_freq = 1
        args.eval_x_steps = 1
        args.eval_y_steps = 1
        args.eval_beg = 0

    return args


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    set_rng_seed(args.seed)
    trainer = Trainer(args)

    player = trainer.init_player(trainer.demos_train[0])

    # video = []
    # from tqdm import tqdm
    # for i in tqdm(range(len(trainer.demos_train[0]["data"]))):
    #     image = trainer.render_single_frame(player, i)
    #     video.append(image.cpu().numpy())

    # video = (np.stack(video) * 255).astype(np.uint8)
    # import imageio
    # imageio.mimsave("raw.mp4", video, fps=120)

    # player = trainer.init_player(trainer.demos_train[20])
    # actions = trainer.replay_demo(player)
    # player = trainer.init_player(trainer.demos_train[20])
    # trainer.validate_actions(player, actions, "test.mp4")

    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    if not args.eval_only:
        trainer.train()
        trainer.load_checkpoint(f"{trainer.log_dir}/model_best.pth", False)

    avg_success = trainer.eval_in_env(trainer.args, "best",
        trainer.args.final_x_steps, trainer.args.final_y_steps)
    print(f"Average success rate: {avg_success:.4f}")
    wandb.log({"final_success": avg_success})

    wandb.finish()


if __name__ == "__main__":
    main()
