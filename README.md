# Intelligent Maze Agent

This project trains an agent to find the exit of randomly generated mazes. The agent learns using Deep Q-Learning (DQN) paired with an LSTM network, relying on a partial observation of the tiles surrounding it. A Pygame window allows you to visualize its movements.

## Contents

- `train.py`: Trains the agent and saves its weights to `model.pth`.
- `visualize.py`: Launches the visualization of the agent using the saved model.
- `agent/`: Contains the Q-network, DQN agent, and replay memory.
- `env/`: Contains the Gymnasium environment and maze generation code.
- `model.pth`: Pre-trained model provided with the project.

## Prerequisites

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`

## Installation (Windows / PowerShell)

From the project root directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script execution/activation, you can install the dependencies directly through the virtual environment's executable without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

To train a new model:

```powershell
python train.py
```

Training runs for 3,000 episodes by default and overwrites `model.pth` upon completion.

To visualize the agent using `model.pth`:

```powershell
python visualize.py
```

Close the maze window to exit the visualization.

## Technical Details

- **Maze Generation:** The maze is regenerated at the start of each episode.
- **Agent Observation:** The agent's observation space is limited to a local 3 × 3 tile view.
- **Action Space:** The available actions are Up, Down, Left, and Right.
- **Dimensions:** The environment requires an odd maze size; training and visualization are configured to a default size of 9.
