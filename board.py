import numpy as np

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
        else:
            self.board = b
            for i in range(4):
                for j in range(4):
                    if self.board[i][j] == 0:
                        self.hole = (i, j)
    def __str__(self):
        return str(self.board)

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

    def perform_action(self, k):
        if k not in self.actions():
            raise ValueError('illegal action')
        (i, j) = self.hole
        (di, dj) = DISPLACEMENTS[k]
        (ii, jj) = (i+di, j+dj)
        self.board[i][j], self.board[ii][jj] = self.board[ii][jj], self.board[i][j]
        self.hole = (ii, jj)
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
        return -dist

    def is_solved(self):
        return (self.board == np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ])).all()