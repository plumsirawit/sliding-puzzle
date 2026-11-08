from board import BoardState
import numpy as np
import random
import queue

class BFSGenerator:
    def __init__(self, walklen=10):
        print('[LOG] BFSGenerator init with walklen', walklen)
        self.walklen = walklen
        self.INITIAL_BOARD = BoardState(np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ]))
        dists = {self.INITIAL_BOARD.hash(): 0}
        q = queue.Queue()
        q.put(self.INITIAL_BOARD)
        while not q.empty():
            board = q.get()
            q.task_done()
            d = dists[board.hash()]
            if d >= walklen:
                continue
            for k in board.actions():
                next_board = board.copy()
                next_board.perform_action(k)
                h = next_board.hash()
                if h not in dists or dists[h] > d+1:
                    dists[h] = d+1
                    q.put(next_board)
        self.BUCKETS = [[] for _ in range(walklen+1)]
        for (k, v) in dists.items():
            self.BUCKETS[v].append(BoardState(k))

bfs_generator = BFSGenerator()

def sample_board(walklen, verbose=False):
    global bfs_generator
    if bfs_generator.walklen < walklen:
        bfs_generator = BFSGenerator(max(bfs_generator.walklen+1, walklen))
    state = random.choice([x for xs in bfs_generator.BUCKETS[1:walklen+1] for x in xs]).copy()
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
