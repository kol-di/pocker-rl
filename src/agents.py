from abc import ABC, abstractmethod
from random import randint, random
from contextlib import contextmanager
from collections import defaultdict
import torch
import torch.optim as optim
import numpy as np

from src.util.loss import WeightedMSE
from src.util.nn import PockerNN



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ACTION_SPACE_SZ = 2

class BaseAgent(ABC):
    def __init__(self, config):
        self._eps = config['EPS']
        self._eps_min = config['EPS_MIN']
        self._eps_step = config['EPS_STEP']
        self._eval = False

    @abstractmethod
    def best_value_and_action(self, state):
        pass

    @abstractmethod
    def value_update(self, *args, **kwargs):
        pass

    def decrease_eps(self):
        if (self._eps - self._eps_step) >= self._eps_min:
            self._eps -= self._eps_step

    @contextmanager
    def eval(self):
        try:
            self._eval = True
            yield
        finally:
            self._eval = False


class ValueTableAgent(BaseAgent):
    def __init__(self, config, *args, **kwargs):
        super(ValueTableAgent, self).__init__(config, *args, **kwargs)

        self.values = defaultdict(float)
        self.alpha = config['VALUE_TABLE_ALPHA']

    def best_value_and_action(self, state):
        best_value, best_action = None, None

        if not self._eval:
            if random() < self._eps:
                best_action = randint(0, ACTION_SPACE_SZ-1)
                best_value = self.values[(state, best_action)]

                return best_value, best_action

        for action in range(ACTION_SPACE_SZ):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_value, best_action

    def value_update(self, state, action, reward):
        old_v = self.values[(state, action)]
        self.values[(state, action)] = old_v * (1-self.alpha) + reward * self.alpha



class NNAgent(BaseAgent):
    def __init__(self, config, loss=WeightedMSE(), *args, **kwargs):
        super(NNAgent, self).__init__(config, *args, **kwargs)
        self.net = PockerNN().to(DEVICE)

        self.optimiser = optim.Adam(self.net.parameters(), lr=0.0001, eps=1e-6)
        self.scheduler = optim.lr_scheduler.ConstantLR(self.optimiser, factor=0.995, total_iters=500)
        self.loss = loss


    def best_value_and_action(self, state):
        best_value, best_action = None, None

        with torch.no_grad():
            if not self._eval:
                if random() < self._eps:
                    best_action = randint(0, ACTION_SPACE_SZ-1)
                    sample_t = torch.tensor(
                        [state.sb_stack, state.bb_stack, state.blind_order, state.card_rk, best_action],
                        dtype=torch.float32
                    ).to(DEVICE)

                    best_value = self.net(sample_t)

                    return best_value, best_action

            for action in range(ACTION_SPACE_SZ):
                sample_t = torch.tensor(
                    [state.sb_stack, state.bb_stack, state.blind_order, state.card_rk, action],
                    dtype=torch.float32
                ).to(DEVICE)

                action_value = self.net(sample_t)
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action

        return best_value.item(), best_action

    def value_update(self, batch, weights=None):
        loss = self.calc_loss(batch, weights=weights)

        self.optimiser.zero_grad()
        loss.backward()
        grads = np.concatenate(
            [p.grad.data.cpu().numpy().flatten()
             for p in self.net.parameters()
             if p.grad is not None])

        self.optimiser.step()
        # self.scheduler.step()

        return grads, loss.item()
    
    def calc_loss(self, batch, weights=None):
        batch_t = torch.tensor(
            [[s.sb_stack, s.bb_stack, s.blind_order, s.card_rk, a, r] for s, a, r in batch],
            dtype=torch.float32
        ).to(DEVICE)

        batch_samples_t = batch_t[:, :5]
        batch_rewards_t = batch_t[:, 5]

        preds = self.net(batch_samples_t).view(-1)

        if weights is not None:
            loss = self.loss(preds, batch_rewards_t, weights=weights)
        else:
            loss = self.loss(preds, batch_rewards_t)

        return loss
