from rfsf.util.constant_lr import ConstantLR


class MockLR(ConstantLR):
    # noinspection PyMissingConstructor
    def __init__(self, *args, **kwargs):
        pass

    def get_last_lr(self, *args, **kwargs):
        return []

    def get_lr(self, *args, **kwargs):
        return []

    def load_state_dict(self, *args, **kwargs):
        pass

    def state_dict(self, *args, **kwargs):
        return {}

    def step(self, *args, **kwargs):
        pass
