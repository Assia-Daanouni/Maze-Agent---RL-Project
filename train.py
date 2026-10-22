import numpy as np
from collections import deque
from env.maze_env import MazeEnv
from agent.dqn import DQNAgent
from agent.replay_buffer import ReplayBuffer

SEQ_LEN = 8

def train(n_episodes=3000, maze_size=9):
    env = MazeEnv(size=maze_size)
    agent = DQNAgent(eps_decay=20000, seq_len=SEQ_LEN)
    buffer = ReplayBuffer(capacity=50_000)

    rewards_history = []
    success_history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()

        # Initialise la séquence avec des zéros + première observation
        obs_seq = deque(
            [np.zeros(9)] * (SEQ_LEN - 1) + [obs],
            maxlen=SEQ_LEN
        )

        total_reward = 0
        done = False

        while not done:
            state_seq = np.array(obs_seq) 
            action = agent.select_action(state_seq)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Nouvelle séquence avec next_obs ajouté
            next_obs_seq = deque(obs_seq, maxlen=SEQ_LEN)
            next_obs_seq.append(next_obs)

            buffer.push(
                np.array(obs_seq),
                action,
                reward,
                np.array(next_obs_seq),
                float(done)
            )
            agent.update(buffer)

            obs_seq.append(next_obs)
            total_reward += reward

        rewards_history.append(total_reward)
        success_history.append(float(terminated))

        if episode % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            eps = agent.eps_end + (agent.eps_start - agent.eps_end) * \
                  np.exp(-agent.steps_done / agent.eps_decay)
            print(f"Ep {episode:4d} | Reward: {avg_r:6.2f} | "
                  f"Success: {success_rate:5.1f}% | ε: {eps:.3f}")

    return agent, rewards_history, success_history

if __name__ == "__main__":
    agent, rewards, successes = train()
    import torch
    torch.save(agent.q_net.state_dict(), "model.pth")
    print("Modèle sauvegardé !")