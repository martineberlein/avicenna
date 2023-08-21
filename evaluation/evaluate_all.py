from typing import List, Tuple, Callable, Dict, Any

from isla.language import ISLaUnparser

from evaluate_calculator import eval_config as calc_config
from evaluate_heartbleed import eval_config as heartbleed_config
from evaluate_middle import eval_config as middle_config
from tests4py_benchmark.evaluate_pysnooper_1 import eval_config as pysnooper1_config
from tests4py_benchmark.evaluate_pysnooper_2 import eval_config as pysnooper2_config
from tests4py_benchmark.evaluate_youtubedl_1 import eval_config as youtubedl1_config
from tests4py_benchmark.evaluate_youtubedl_2 import eval_config as youtubedl2_config
from tests4py_benchmark.evaluate_youtubedl_3 import eval_config as youtubedl3_config


from avicenna.evaluation_setup import (
    generate_evaluation_data_set,
    EvaluationResult,
    evaluate_diagnosis,
)

from avicenna.avicenna import Avicenna

if __name__ == "__main__":
    DEFAULT_PARAM = {"max_iterations": 100, "timeout": 3600, "log": False}

    subjects: List[Tuple[str, Callable]] = [
        ("Calculator", calc_config),
        ("Heartbleed", heartbleed_config),
        ("Pysnooper_1", pysnooper1_config),
        ("Pysnooper_2", pysnooper2_config),
        ("YoutubeDL_1", youtubedl1_config),
        ("YoutubeDL_2", youtubedl2_config),
        ("YoutubeDL_3", youtubedl3_config),
        ("Middle", middle_config)
    ]

    results = dict()
    for subject in subjects:
        name, config = subject
        config_param: Dict[str, Any] = config()
        param = DEFAULT_PARAM.copy()
        param.update(config_param)

        # measure time
        avicenna = Avicenna(**param)
        diagnosis = avicenna.explain()
        results[name] = EvaluationResult(name, diagnosis[0])

    for subject in subjects:
        name, config = subject
        param = config()
        grammar = param["grammar"]
        oracle = param["oracle"]
        initial_inputs = param["initial_inputs"]

        eval_data = generate_evaluation_data_set(grammar, oracle, initial_inputs, 100)
        results[name].evaluation_data = eval_data
        evaluate_diagnosis(grammar, results[name])

    for subject in subjects:
        name, _ = subject
        result = results[name]
        print(f"Evaluation Result for the {name} Subject:")
        print(f"Learned failure-inducing Diagnosis:")
        print(f"{ISLaUnparser(result.formula).unparse()}")
        print(f"Archived a precision of {result.precision*100}%")
        print(f"Archived a recall of {result.recall*100}%", end="\n\n")
