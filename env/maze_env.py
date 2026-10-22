import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, size=11, render_mode=None):
        super().__init__()
        assert size % 2 == 1, "size doit être impair (ex: 11, 13, 15)"
        self.size = size
        self.render_mode = render_mode

        # Action space: 4 directions (haut, bas, gauche, droite)
        self.action_space = spaces.Discrete(4)

        # Observation: vue partielle 3x3 autour du robot (9 cases)
        # Chaque case: 0=libre, 1=mur, 2=sortie
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(9,), dtype=np.float32
        )

        self._action_to_delta = {
            0: (-1, 0),  # haut
            1: (1, 0),   # bas
            2: (0, -1),  # gauche
            3: (0, 1),   # droite
        }

    def _generate_maze(self):
        """Recursive Backtracking DFS — garantit maze résoluble"""
        maze = np.ones((self.size, self.size), dtype=np.int32)
        
        def carve(r, c):
            dirs = [(0,2),(0,-2),(2,0),(-2,0)]
            np.random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 < nr < self.size-1 and 0 < nc < self.size-1:
                    if maze[nr, nc] == 1:
                        maze[r+dr//2, c+dc//2] = 0  # casse le mur
                        maze[nr, nc] = 0
                        carve(nr, nc)

        maze[1, 1] = 0
        carve(1, 1)
        return maze
    def _get_obs(self):
        """Vue 3x3 autour du robot — observation partielle (POMDP)"""
        r, c = self.agent_pos
        obs = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) == self.goal_pos:
                        obs.append(2.0)   # sortie visible !
                    else:
                        obs.append(float(self.maze[nr, nc]))
                else:
                    obs.append(1.0)  # hors limites = mur
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, old_pos, new_pos, hit_wall):
        """Reward shaping clé pour accélérer l'apprentissage"""
        if new_pos == self.goal_pos:
            return +10.0                  # objectif atteint !
        if hit_wall:
            return -0.5                   # pénalité mur
        
        # Reward basé sur la distance Manhattan (shaping)
        old_dist = abs(old_pos[0]-self.goal_pos[0]) + \
                   abs(old_pos[1]-self.goal_pos[1])
        new_dist = abs(new_pos[0]-self.goal_pos[0]) + \
                   abs(new_pos[1]-self.goal_pos[1])
        
        shaping = 0.1 * (old_dist - new_dist)  # + si on se rapproche
        return -0.01 + shaping                  # pénalité par step
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.maze = self._generate_maze()   # nouveau maze à chaque épisode !
        self.agent_pos = (1, 1)             # départ fixe
        self.goal_pos = (self.size-2, self.size-2)  # sortie en bas-droite
        self.maze[self.goal_pos] = 0
        self.steps = 0
        self.max_steps = self.size * self.size * 2
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        r, c = self.agent_pos
        dr, dc = self._action_to_delta[action]
        nr, nc = r+dr, c+dc

        hit_wall = False
        if self.maze[nr, nc] == 1:      # mur — on reste sur place
            hit_wall = True
            nr, nc = r, c

        old_pos = self.agent_pos
        self.agent_pos = (nr, nc)

        reward = self._compute_reward(old_pos, self.agent_pos, hit_wall)
        terminated = (self.agent_pos == self.goal_pos)
        truncated  = (self.steps >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, {}
    