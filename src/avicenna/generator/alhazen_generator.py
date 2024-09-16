from avicenna.data import Input
from .generator import Generator


class AlhazenGenerator(Generator):
    """
    The Alhazen Generator: A generator that uses decision trees to generate new inputs.
    """

    def generate(self, *args, **kwargs) -> Input:
        pass
