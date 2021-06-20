from rfsf.util.constant_lr import ConstantLR


class MockLR(ConstantLR):
    """Mock learning rate scheduler that does nothing while all methods accept anything."""

    # Do not invoke the super constructor to prevent any behavior.
    # noinspection PyMissingConstructor
    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called,unused-argument
        pass

    def get_last_lr(self, *args, **kwargs):  # pylint: disable=unused-argument
        return []

    def get_lr(self, *args, **kwargs):  # pylint: disable=unused-argument
        return []

    def load_state_dict(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass

    def state_dict(self, *args, **kwargs):  # pylint: disable=unused-argument
        return {}

    def step(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass
