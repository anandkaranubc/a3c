# A3C for Kung Fu

![A3C Results](A3C_Results.gif)

This project implements the Asynchronous Advantage Actor-Critic (A3C) algorithm to play the classic Atari game Kung Fu. The implementation leverages PyTorch for building and training the neural network and Gymnasium for the environment.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Neural Network Architecture](#neural-network-architecture)
- [Training Process](#training-process)
- [Acknowledgments](#acknowledgments)

## Installation

### Required Packages

Before running the project, ensure you have the necessary packages installed. You can install them using the following commands:

```sh
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
apt-get install -y swig
pip install gymnasium[box2d]
pip install opencv-python
pip install torch
```

### Importing Libraries

The following libraries are required and imported in the script:

```python
import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
```

## Project Structure

The project is divided into several parts for clarity and organization:

1. **Installing Required Packages and Importing Libraries**
2. **Building the AI (Neural Network Architecture)**
3. **Training the AI (Setting Up the Environment and Training Loop)**

## Usage

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/a3c_for_kung_fu.git
    cd a3c_for_kung_fu
    ```

2. **Run the script:**

    ```sh
    python a3c_for_kung_fu.py
    ```

## Neural Network Architecture

The neural network for this project is built using PyTorch. It consists of convolutional layers followed by fully connected layers to output the action values and state value.

### Network Class

```python
class Network(nn.Module):

    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=512, out_features=128)
        self.fc2a = torch.nn.Linear(in_features=128, out_features=action_size)
        self.fc2s = torch.nn.Linear(in_features=128, out_features=1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0]
        return action_values, state_value
```

## Training Process

The training process involves setting up the environment, preprocessing the observations, and running the training loop using multiple processes to leverage parallelism.

### Setting Up the Environment

```python
class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        super(PreprocessAtari, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)

    def observation(self, obs):
        return PreprocessAtari.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img.mean(2)
        img = img / 255.0
        img = cv2.resize(img, (84, 110))
        img = img[13:97, :]
        img = np.reshape(img, [1, 84, 84])
        return img
```

### Training Loop

The training loop leverages multiprocessing for running parallel agents. Each agent updates the global network asynchronously.

```python
def train(rank, params, shared_model, counter, lock, optimizer):
    torch.manual_seed(params.seed + rank)
    env = PreprocessAtari(gym.make(params.env_name))
    model = Network(env.action_space.n)
    model.train()

    state = env.reset()
    state = torch.from_numpy(state)

    while True:
        # Training logic here
        pass
```

## Acknowledgments

This project is inspired by the A3C algorithm presented in "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al. It also leverages the Gymnasium library for environment setup and PyTorch for building and training the neural network.
