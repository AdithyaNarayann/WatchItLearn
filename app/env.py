from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from app.config import MAZE_ROWS, MAZE_COLS, NUM_STATES, NUM_ACTIONS, ACTIONS, MAZE_ASCII, REWARD_STEP, REWARD_GOAL

def encode(r: int, c: int) -> int:
    return r * MAZE_COLS + c

def decode(s: int) -> Tuple[int, int]:
    return (s // MAZE_COLS, s % MAZE_COLS)

class MazeWorld:

    def __init__(self, layout: Optional[List[str]]=None):
        self.grid: List[str] = layout or MAZE_ASCII
        self.rows = MAZE_ROWS
        self.cols = MAZE_COLS
        self.start_pos: Tuple[int, int] = (1, 1)
        self.goal_pos: Tuple[int, int] = (9, 9)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 'S':
                    self.start_pos = (r, c)
                elif self.grid[r][c] == 'G':
                    self.goal_pos = (r, c)
        self.start_state = encode(*self.start_pos)
        self.goal_state = encode(*self.goal_pos)
        self.next_state = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int32)
        self.reward = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)
        self.done = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=bool)
        self._build_tables()
        self.optimal_len, self.optimal_path = self._bfs()

    def _build_tables(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                s = encode(r, c)
                tile = self.grid[r][c]
                if (r, c) == self.goal_pos:
                    for a in range(NUM_ACTIONS):
                        self.next_state[s, a] = s
                        self.reward[s, a] = 0.0
                        self.done[s, a] = True
                    continue
                for a, (dr, dc) in enumerate(ACTIONS):
                    nr, nc = (r + dr, c + dc)
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.grid[nr][nc] == '#':
                        self.next_state[s, a] = s
                        self.reward[s, a] = REWARD_STEP
                        self.done[s, a] = False
                    elif (nr, nc) == self.goal_pos:
                        self.next_state[s, a] = encode(nr, nc)
                        self.reward[s, a] = REWARD_GOAL
                        self.done[s, a] = True
                    else:
                        self.next_state[s, a] = encode(nr, nc)
                        self.reward[s, a] = REWARD_STEP
                        self.done[s, a] = False

    def _bfs(self) -> Tuple[int, List[Tuple[int, int]]]:
        q = deque([(self.start_pos[0], self.start_pos[1], 0, [self.start_pos])])
        visited = {self.start_pos}
        while q:
            r, c, dist, path = q.popleft()
            if (r, c) == self.goal_pos:
                return (dist, path)
            for dr, dc in ACTIONS:
                nr, nc = (r + dr, c + dc)
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr][nc] != '#' and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc, dist + 1, path + [(nr, nc)]))
        return (-1, [])

    def reset(self) -> int:
        return self.start_state

    def step(self, s: int, a: int) -> Tuple[int, float, bool]:
        return (int(self.next_state[s, a]), float(self.reward[s, a]), bool(self.done[s, a]))

    def set_layout(self, layout: List[str]) -> None:
        self.grid = list(layout)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 'S':
                    self.start_pos = (r, c)
                elif self.grid[r][c] == 'G':
                    self.goal_pos = (r, c)
        self.start_state = encode(*self.start_pos)
        self.goal_state = encode(*self.goal_pos)
        self._build_tables()
        self.optimal_len, self.optimal_path = self._bfs()
