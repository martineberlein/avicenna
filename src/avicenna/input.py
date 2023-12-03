from typing import Optional

from isla.derivation_tree import DerivationTree

from debugging_framework.oracle import OracleResult
from debugging_framework.input import Input as TestInput
from avicenna.features import FeatureVector


class Input(TestInput):
    """
    Class describing a test input.
    """

    def __init__(self, tree: DerivationTree, oracle: OracleResult = None):
        super().__init__(tree, oracle)
        self.__features: Optional[FeatureVector] = None

    @property
    def features(self) -> FeatureVector:
        return self.__features

    @features.setter
    def features(self, features_: FeatureVector):
        self.__features = features_

    def update_features(self, features_: FeatureVector) -> "Input":
        self.__features = features_
        return self
