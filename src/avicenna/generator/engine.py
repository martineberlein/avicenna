from typing import List, Set, Type
from queue import Queue, Empty
from threading import Thread
from multiprocessing import Process, Queue as ProcessQueue, Manager

from abc import ABC, abstractmethod

from avicenna.learning.table import Candidate
from avicenna.generator.generator import Generator, ISLaSolverGenerator, ISLaGrammarBasedGenerator

from debugging_benchmark.calculator.calculator import calculator_grammar


class Engine:

    def __init__(
            self,
            generator: Generator,
            workers: int = 20,
    ):
        self.workers = [
            generator
            for _ in range(workers)
        ]

    def generate(self, candidates: List[Candidate]):
        pass


class ParallelEngine(Engine):

    def generate(self, candidates: List[Candidate]):
        """
        Generate new inputs for the given candidates in parallel.
        :param List[Candidate] candidates: The candidates to generate new inputs for.
        :return:
        """

        threads = []
        candidate_queue = Queue()
        output_queue = Queue()
        for candidate in candidates:
            candidate_queue.put(candidate)
        for worker in self.workers:
            thread = Thread(target=worker.run_with_engine, args=(candidate_queue, output_queue))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        test_inputs = set()
        while not output_queue.empty():
            test_inputs.update(output_queue.get())

        return test_inputs


class ProcessBasedParallelEngine(Engine):
    def generate(self, candidates: List[Candidate]):
        """
            Generate new inputs for the given candidates in parallel.
            :param List[Candidate] candidates: The candidates to generate new inputs for.
            :return:
            """

        processes = []
        candidate_queue = ProcessQueue()
        manager = Manager()
        output_list = manager.list()  # Using Manager list to share data between processes

        for candidate in candidates:
            candidate_queue.put(candidate)

        for worker in self.workers:
            process = Process(target=worker.run_with_engine, args=(candidate_queue, output_list))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        test_inputs = set()
        for output in output_list:
            test_inputs.update(output)
        return test_inputs


if __name__ == "__main__":
    from isla import language

    formula1 = """exists <function> elem_0 in start:
        (= elem_0 "cos")
    """
    formula1 = language.parse_isla(formula1)
    formula2 = """exists <function> elem_0 in start:
        (= elem_0 "sqrt")
    """
    formula2 = language.parse_isla(formula2)

    candidate1 = Candidate(formula=formula1)
    candidate2 = Candidate(formula=formula2)

    gen = ISLaGrammarBasedGenerator(calculator_grammar)
    engine = ParallelEngine(gen, workers=3)
    test_inputs = engine.generate([candidate1, candidate2, candidate2])
    print("Generated inputs: ", len(test_inputs))

    gen = ISLaSolverGenerator(calculator_grammar)
    process_engine = ProcessBasedParallelEngine(gen, workers=5)
    test_inputs = process_engine.generate([candidate1, candidate1, candidate1, candidate1, candidate1])
    print("Generated inputs: ", len(test_inputs))
    for inp in test_inputs:
        print(inp)

    print("done")
