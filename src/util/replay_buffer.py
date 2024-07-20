from collections import deque
import warnings
from random import sample
import numpy as np


class ReplayBuffer:
    def __init__(self, config):
        self._maxlen = config['REPLAY_BUFFER_LEN']
        self._buffer = deque(maxlen=self._maxlen)
        self.accum_len = 0

    def add(self, el):
        self._buffer.append(el)
        self.accum_len += 1

    def extend(self, els):
        self._buffer.extend(els)
        self.accum_len += len(els)

    def len(self):
        return len(self._buffer)

    def full(self):
        return self.len() >= self._maxlen

    def sample(self, sample_size):
        if self.len() < self._maxlen:
            warnings.warn("Sampling from buffer which is not full")
        return sample(list(self._buffer), sample_size)
    


class PrioReplayBuffer(ReplayBuffer):
    def __init__(self, config, *args, **kwargs):
        super(PrioReplayBuffer, self).__init__(config, *args, **kwargs)

        self._prio = deque(maxlen=self._maxlen)
        self.prob_alpha = config['PRIO_BUFFER_ALPHA']
        self.prob_beta = config['PRIO_BUFFER_BETA']
        self.prob_beta_step = config['PRIO_BUFFER_BETA_STEP']

    
    def add(self, el):
        max_prio = max(self._prio) if self.len() else 1.0
        self._prio.append(max_prio)

        super(PrioReplayBuffer, self).add(el)

    def extend(self, els):
        max_prio = max(self._prio) if self.len() else 1.0
        prios = [max_prio] * len(els)
        self._prio.extend(prios)

        super(PrioReplayBuffer, self).extend(els)

    def sample(self, sample_size):
        probs = np.array(self._prio) ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(self.len(), sample_size, p=probs)
        samples = [self._buffer[idx] for idx in indices]

        weights = (self.len() * probs[indices]) ** (-self.prob_beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self._prio[idx] = prio

    def update_beta(self):
        self.prob_beta = min(1.0, self.prob_beta + self.prob_beta_step)
