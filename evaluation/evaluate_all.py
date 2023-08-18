from typing import List, Tuple, Callable, Dict, Any

from isla.language import ISLaUnparser
from example_calculator import eval_config as calc_config
from example_heartbleed import eval_config as heartbleed_config
from example_middle import eval_config as middle_config
from evaluation_setup import generate_evaluation_data_set

from avicenna.avicenna import Avicenna

if __name__ == "__main__":
    default_param = {
        "max_iterations": 10,
        "timeout": 3600,
        "log": False
    }

    subjects: List[Tuple[str, Callable]] = [
        ("Calculator", calc_config),
        ("Heartbleed", heartbleed_config),
        ("Middle", middle_config)
    ]

    results = dict()
    for subject in subjects:
        name, config = subject
        config_param: Dict[str, Any] = config()
        default_param.update(config_param)
        print(default_param)

        # measure time
        avicenna = Avicenna(**default_param)
        diagnosis = avicenna.explain()
        print("Final Diagnosis:")
        print(ISLaUnparser(diagnosis[0]).unparse())
        results[name] = diagnosis[0]

    # for subject in subjects:
    #     name, config = subject
    #     param = config_param
    #     grammar = config["grammar"]
    #
    #     eval_data = generate_evaluation_data_set()