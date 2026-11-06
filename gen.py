from board import BoardState
import numpy as np
import random

def sample_board(walklen, verbose=False):
    state = BoardState(np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]))
    chosen_dist = random.randint(0, walklen)
    for i in range(chosen_dist):
        k = random.choice(state.actions())
        state.perform_action(k)
    while state.is_solved():
        k = random.choice(state.actions())
        state.perform_action(k)
    if verbose:
        print('[LOG] Board Generated:')
        print(state)
    return state

def main(args):
    state = sample_board(args.walklen, args.verbose)
    outst = '\n'.join([' '.join(list(map(str, state.board[i]))) for i in range(4)])
    if len(args.out) == 0:
        print(outst)
    else:
        with open(args.out, 'w') as f:
            f.write(outst)
            f.write('\n')
    print('[LOG] Done!')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--walklen', type=int, default=10)
    parser.add_argument('--out', type=str, default='')
    args = parser.parse_args()
    print('[LOG] Generating a board. Walk length =', args.walklen)
    main(args)
