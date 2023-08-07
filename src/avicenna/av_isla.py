from isla.solver import ISLaSolver, DerivationTree


class AvicennaISlaSolver(ISLaSolver):
    def __int__(self, grammar):
        pass

    def solve(self) -> DerivationTree:
        print("ey")


if __name__ == "__main__":
    av = AvicennaISlaSolver("f")

    # av.solve()
