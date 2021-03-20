import numpy as np
from collections import namedtuple, deque


class experienceReplayBuffer:

    def __init__(self, memory_size=5000):
        self.memory_size = memory_size
        self.Buffer = namedtuple('Buffer', field_names=['state', 'action', 'reward'])
        self.replay_memory = deque(maxlen=memory_size)

    # Получаем Батч случайных примеров из буфера памяти
    def sample_batch(self, batch_size=64):
        if batch_size>len(self.replay_memory):
            batch_size = len(self.replay_memory)
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    # Получаем Батч случайных примеров из буфера памяти
    def all_random_sample_batch(self):
        samples = np.random.choice(len(self.replay_memory), len(self.replay_memory), replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    # Получаем Батч случайных примеров из буфера памяти
    def all_sample_batch(self):
        batch = zip(*self.replay_memory)
        return batch

    # Добавление одиночного примера в буфер памяти
    def append(self, state, action, reward):
        self.replay_memory.append(self.Buffer(state, action, reward))

    # Добавление Батча примеров в буфер памяти
    def append_butch(self, state_batch, action_batch, reward_batch):
        for i in range(state_batch.shape[0]):
            self.append(state_batch[i], action_batch[i], reward_batch[i])

    # Очистка буфера памяти
    def clear(self):
        self.replay_memory.clear()
        return

    # Расчет длины буфера памяти с датасетом
    def print_len(self):
        return len(self.replay_memory)
