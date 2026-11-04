from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel
MAZE_ROWS = 11
MAZE_COLS = 11
NUM_STATES = MAZE_ROWS * MAZE_COLS
NUM_ACTIONS = 4
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
MAZE_PRESET_LABYRINTH: List[str] = ['###########', '#S....#...#', '#.#####.#.#', '#.#...#.#.#', '#.#.#.#.#.#', '#.#.#...#.#', '#.#.#.#.#.#', '#...#.###.#', '#########.#', '#........G#', '###########']
MAZE_PRESET_CLASSIC: List[str] = ['###########', '#S....#...#', '#.#####.#.#', '#.#.#...#.#', '#.#.#.#.#.#', '#.#...#.#.#', '#.#####.#.#', '#.......#.#', '#########.#', '#........G#', '###########']
MAZE_PRESET_ZIGZAG: List[str] = ['###########', '#S....#...#', '#.#####.#.#', '#...#...#.#', '###.#.#.#.#', '#...#.#.#.#', '#.#####.#.#', '#.......#.#', '#########.#', '#........G#', '###########']
MAZE_PRESETS: List[List[str]] = [MAZE_PRESET_LABYRINTH, MAZE_PRESET_CLASSIC, MAZE_PRESET_ZIGZAG]
MAZE_PRESET_NAMES: List[str] = ['Labyrinth (32)', 'Classic (28)', 'Zig-Zag (32)']
MAZE_ASCII: List[str] = MAZE_PRESET_LABYRINTH
REWARD_STEP = -0.01
REWARD_GOAL = 1.0
MAX_ATTEMPT_STEPS = 200
ALPHA = 0.5
GAMMA = 0.97
TARGET_SECONDS = 40.0
EXPLORE_RATIO = 0.7
EPSILON_START = 1.0
EPSILON_FLOOR = 0.02
PHASE1_MAX_ATTEMPTS = 5
PHASE1_STEPS_PER_SEC = 10.0
TURBO_FPS = 25.0
FINAL_WALK_STEPS_PER_SEC = 12.0
CONVERGENCE_CHECK_INTERVAL = 25
CONVERGENCE_ROLLOUTS = 10
CONVERGENCE_STREAK_REQUIRED = 3
MAX_TIMEOUT_SECONDS = TARGET_SECONDS * 1.5
StageType = Literal['idle', 'exploring', 'learning', 'converging', 'solved']

class ResultData(BaseModel):
    steps: int
    optimal: int

class ServerFrame(BaseModel):
    stage: StageType
    t: float
    T: float
    attempt: int
    random_pct: int
    agent: Tuple[int, int]
    trail: List[Tuple[int, int]]
    value: List[List[float]]
    best: List[List[int]]
    steps_history: List[int]
    result: Optional[ResultData] = None
    best_steps: Optional[int] = None
    total_transitions: int = 0
    optimal_steps: int = 32
    maze_index: int = 0
    grid: List[str] = []
