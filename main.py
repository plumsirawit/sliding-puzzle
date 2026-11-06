from trainer import Trainer
from board import BoardState
import torch
import numpy as np

def validate(trainer, state, episode_limit=1000, verbose=False):
    turns = 0
    total_reward = 0.0
    while not state.is_solved() and turns < episode_limit:
        allowed = state.actions()
        mask = [0 if i not in allowed else 1 for i in range(4)]
        cat = trainer.policy.run_policy(state.to_tensor(), torch.as_tensor(mask, dtype=torch.float32))
        action = cat.sample().item()
        if verbose:
            print('[LOG] Current state is')
            print(state)
            print('[LOG] Sampled action is', 'ULDR'[action])
        reward = state.perform_action(action)
        total_reward += reward
        turns += 1
    return (state.is_solved(), turns, total_reward)

def main(args):
    print('[LOG] Starting main loop. verbose=%s. lr=%f.' % (str(args.verbose), args.lr))
    TEST_BOARDS = []
    for tc in range(1, 11):
        fname = 'data/' + str(tc).zfill(4) + '.txt'
        with open(fname) as f:
            blist = list(map(lambda line : list(map(int, line.split())), f.readlines()))
            print('[DBG]', blist)
            board = BoardState(np.array(blist))
            TEST_BOARDS.append(board)
    trainer = Trainer(args.lr)
    for epoch in range(1, 251):
        trainer.train_one_epoch(5000, 100, args.verbose)
        if epoch % 10 == 0:
            trainer.MAX_SCAFFOLD_DIST += 1
        if epoch % 50 == 0:
            # validate against the freezed testcases
            solved_count = 0
            total_turns = 0
            total_reward = 0.0
            for board in TEST_BOARDS:
                (is_solved, turns, reward) = validate(trainer, board.copy(), verbose=args.verbose)
                if is_solved:
                    solved_count += 1
                total_turns += turns
                total_reward += reward
            print('[LOG] epoch %d valid. solved: %d/10 \t avg turns: %f \t avg reward: %f' % (epoch, solved_count, total_turns/10, total_reward/10))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    main(args)
