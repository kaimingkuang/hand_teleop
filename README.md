# Usage
```bash
python train.py --demo-folder=sim/raw_data/pick_place_mustard_bottle_aug \
    --seed=42 \
    --eval-freq=1 \
    --eval-beg=0 \
    --wd-coef=1e-4 \
    --lr=1e-3 \
    --debug \
    --wandb-off
    --batch-size=64
```

# Rendering function
trainer.py L204 render_single_frame
trainer.py L268+L309 调用render_single_frame