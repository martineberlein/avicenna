import unittest

from avicenna.alhazen import Alhazen
from avicenna.data import Input, OracleResult

from avicenna.learning.alhazen_learner import ModelTrainer, DecisionTreeModel
from avicenna.features.feature_collector import GrammarFeatureCollector
from avicenna.features.features import Feature, NumericFeature, ExistenceFeature, DerivationFeature

from resources.subjects import get_calculator_subject



class AlhazenTestCase(unittest.TestCase):

    def test_something(self):
        calculator = get_calculator_subject()

        alhazen = Alhazen(
            grammar=calculator.get_grammar(),
            oracle=calculator.get_oracle(),
            initial_inputs=calculator.get_initial_inputs(),
        )

        explanation = alhazen.explain()
        print(explanation)

    def test_decision_tree_learner(self):
        calculator = get_calculator_subject()
        collector = GrammarFeatureCollector(calculator.get_grammar(), feature_types=[NumericFeature, ExistenceFeature, DerivationFeature])

        test_inputs = set(
            [
                Input.from_str(calculator.get_grammar(), inp, inp_oracle)
                for inp, inp_oracle in [
                    ("sqrt(-901)", OracleResult.FAILING),
                    ("sqrt(-8)", OracleResult.FAILING),
                    ("sqrt(10)", OracleResult.PASSING),
                    ("cos(1)", OracleResult.PASSING),
                    ("sin(99)", OracleResult.PASSING),
                    ("tan(-20)", OracleResult.PASSING),
                    ("sqrt(-20)", OracleResult.FAILING),
                ]
            ]
        )
        for inp in test_inputs:
            if not inp.features:
                inp.features = collector.collect_features(inp)

        model_trainer = ModelTrainer()
        decision_tree_model = DecisionTreeModel()
        result = model_trainer.train_model(decision_tree_model, test_inputs)

        # Prepare feature names and class names
        feature_names = model_trainer.data_handler.x_data.columns.tolist()
        class_names = [str(cls) for cls in model_trainer.data_handler.y_data.unique()]

        model_trainer.visualize_decision_tree(feature_names=feature_names, class_names=class_names)


if __name__ == '__main__':
    unittest.main()
