from trainer import Trainer

def main(args):
    trainer = Trainer(args.lr)
    for epoch in range(100):
        trainer.train_one_epoch(1, 100, True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    main(args)
