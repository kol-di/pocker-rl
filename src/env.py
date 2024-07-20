from collections import namedtuple
from random import randint


fields = ('sb_stack', 'bb_stack', 'blind_order', 'card_rk')
round_state = namedtuple('round_state', fields, defaults=(None,) * len(fields))


class Env:
    def __init__(self):
        self.stack_min = 10
        self.stack_max = 50

    def random_start_state(self):
        stack = randint(self.stack_min, self.stack_max)
        blind_order = randint(0, 1)
        return round_state(sb_stack=stack, bb_stack=stack, blind_order=blind_order)