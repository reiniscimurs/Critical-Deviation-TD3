import random
from collections import deque
import numpy as np


class ReplayBuffer(object):
    """
    Standard experience replay buffer for off-policy reinforcement learning algorithms.

    Stores tuples of (state, action, reward, done, next_state) up to a fixed capacity,
    enabling sampling of uncorrelated mini-batches for training.
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        Initialize the replay buffer.

        Args:
            buffer_size (int): Maximum number of transitions to store in the buffer.
            random_seed (int): Seed for random number generation.
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        """
        Add a transition to the buffer.

        Args:
            s (np.ndarray): State.
            a (np.ndarray): Action.
            r (float): Reward.
            t (bool): Done flag (True if episode ended).
            s2 (np.ndarray): Next state.
        """
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """
        Get the number of elements currently in the buffer.

        Returns:
            (int): Current buffer size.
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            (Tuple of np.ndarrays): Batches of states, actions, rewards, done flags, and next states.
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def return_buffer(self):
        """
        Return the entire buffer contents as separate arrays.

        Returns:
            (Tuple of np.ndarrays): Full arrays of states, actions, rewards, done flags, and next states.
        """
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])

        return s, a, r, t, s2

    def clear(self):
        """
        Clear all contents of the buffer.
        """
        self.buffer.clear()
        self.count = 0
