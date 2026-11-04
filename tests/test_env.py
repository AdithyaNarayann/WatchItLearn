import pytest
from app.env import MazeWorld, encode, decode
from app.config import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, REWARD_STEP, REWARD_GOAL
HAND_VERIFIED_PATH = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (7, 2), (7, 3), (6, 3), (5, 3), (4, 3), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 6), (5, 7), (4, 7), (3, 7), (2, 7), (1, 7), (1, 8), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]

def test_bfs_optimum_equals_hand_verified_path():
    env = MazeWorld()
    assert env.optimal_len == 32, f'Expected BFS optimal length 32, got {env.optimal_len}'
    assert len(HAND_VERIFIED_PATH) - 1 == 32
    assert env.optimal_path == HAND_VERIFIED_PATH, 'BFS path must match hand-verified coordinates'
    assert env.start_pos == (1, 1)
    assert env.goal_pos == (9, 9)

def test_wall_collision_wastes_step():
    env = MazeWorld()
    start_s = env.reset()
    next_s, reward, done = env.step(start_s, ACTION_UP)
    assert next_s == start_s
    assert reward == pytest.approx(REWARD_STEP)
    assert done is False
    next_s, reward, done = env.step(start_s, ACTION_LEFT)
    assert next_s == start_s
    assert reward == pytest.approx(REWARD_STEP)
    assert done is False

def test_goal_terminal_reward():
    env = MazeWorld()
    s_before_goal = encode(8, 9)
    next_s, reward, done = env.step(s_before_goal, ACTION_DOWN)
    assert next_s == env.goal_state
    assert reward == pytest.approx(REWARD_GOAL)
    assert done is True
