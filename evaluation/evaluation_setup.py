from typing import List
from avicenna.generator import MutationBasedGenerator
from avicenna.oracle import OracleResult
from avicenna.input import Input


def generate_evaluation_data_set(initial_inputs, grammar, oracle, targeted_size) -> List:
    seed_inputs = set()
    for inp in initial_inputs:
        seed_inputs.add(Input.from_str(grammar, inp, None))

    positive_samples = set()
    negative_samples = set()

    while not (len(positive_samples) > targeted_size and len(negative_samples) > targeted_size):

        mutation_fuzzer = MutationBasedGenerator(
            grammar,
            oracle=oracle,
            seed=seed_inputs,
            yield_negative=True
        )
        result = mutation_fuzzer.generate()
        if result.is_just():
            inp = result.value()
            oracle_result = oracle(inp)
            if oracle_result == OracleResult.BUG:
                positive_samples.add(inp.update_oracle(OracleResult.BUG))
            elif oracle_result == OracleResult.NO_BUG:
                negative_samples.add(inp.update_oracle(OracleResult.NO_BUG))

    return list(positive_samples)[:targeted_size] + list(negative_samples)[:targeted_size]


# if __name__ == "__main__":
#     from example_calculator import grammar, initial_inputs, oracle
#
#     # l = generate_evaluation_data_set(initial_inputs, grammar, oracle, 100)
#
#     # from example_heartbleed import grammar, initial_inputs, oracle
#
#     l = generate_evaluation_data_set(initial_inputs, grammar, oracle, 100)
#
#     for inp in l:
#         print(str(inp), inp.oracle)

