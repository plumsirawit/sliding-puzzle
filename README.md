# sliding-puzzle
A playground for my first RL attempt

## PPO checkpoints

Training saves the actor, critic, both optimizer states, curriculum position,
PPO settings, and random-number-generator states every 50 epochs under
`checkpoints/`.

Resume from a saved checkpoint with:

```powershell
python main.py --resume checkpoints/checkpoint_epoch_0050.pt
```

By default, resumed training continues through epoch 1000 and saves later
checkpoints in `checkpoints/`. Use `--checkpoint-dir` to choose another output
directory.
