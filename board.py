import numpy as np
import torch

DISPLACEMENTS = [(-1, 0), (0, -1), (1, 0), (0, 1)]

class BoardState:
    def __init__(self, b=None):
        if b is None:
            self.board = np.array([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0]
            ])
            self.hole = (3, 3)
        elif isinstance(b, int):
            # from hash
            self.board = np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ])
            for i in range(16):
                self.board[i // 4][i % 4] = b % 16
                b //= 16
            for i in range(4):
                for j in range(4):
                    if self.board[i][j] == 0:
                        self.hole = (i, j)
        else:
            self.board = b
            for i in range(4):
                for j in range(4):
                    if self.board[i][j] == 0:
                        self.hole = (i, j)
        self.dist = self.compute_dist()

    def __str__(self):
        return str(self.board)

    def hash(self):
        # Don't incoroporate into __hash__ (yet)
        # because this is still mutable (especially by perform_action).
        ret = 0
        for i in range(16):
            ret += int(self.board[i // 4][i % 4]) * (16 ** i)
        return ret

    def copy(self):
        return BoardState(self.board.copy())
    
    def actions(self):
        out = []
        (i, j) = self.hole
        for k in range(4):
            (di, dj) = DISPLACEMENTS[k]
            if i+di < 0 or i+di >= 4:
                continue
            if j+dj < 0 or j+dj >= 4:
                continue
            out.append(k)
        return out

    def compute_dist(self):
        dist = 0
        bck = [0] * 16
        for i in range(4):
            for j in range(4):
                bck[self.board[i][j]] = (i, j)
        for i in range(4):
            for j in range(4):
                if i == 3 and j == 3:
                    break
                num = i*4+j+1
                (ii, jj) = bck[num]
                dist += (i - ii)**2 + (j - jj)**2
        return dist

    def perform_action(self, k):
        if k not in self.actions():
            raise ValueError('illegal action')
        (i, j) = self.hole
        (di, dj) = DISPLACEMENTS[k]
        (ii, jj) = (i+di, j+dj)
        self.board[i][j], self.board[ii][jj] = self.board[ii][jj], self.board[i][j]
        self.hole = (ii, jj)
        new_dist = self.compute_dist()
        diff = self.dist - new_dist
        self.dist = new_dist
        return -1 + 0.1 * diff + (100 if self.is_solved() else 0)

    def is_solved(self):
        return (self.board == np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ])).all()

    def to_tensor(self):
        return torch.tensor([1 if self.board[i//4][i%4] == j else 0 for i in range(16) for j in range(16)], dtype=torch.float32)