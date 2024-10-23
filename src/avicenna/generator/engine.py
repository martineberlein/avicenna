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
        self.generator = generator
        self.workers = [
            generator
            for _ in range(workers)
        ]
        self._check_generator_compatability()

    def _check_generator_compatability(self):
        pass

    def generate(self, candidates: List[Candidate]):
        pass


class SingleEngine(Engine):

    def generate(self, candidates: List[Candidate]):
        """
        Generate new inputs for the given candidates.
        :param List[Candidate] candidates: The candidates to generate new inputs for.
        :return:
        """
        test_inputs = set()
        for candidate in candidates:
            test_inputs.update(self.generator.generate_test_inputs(num_inputs=2, candidate=candidate))
        return test_inputs


class ParallelEngine(Engine):

    def _check_generator_compatability(self):
        if isinstance(self.generator, ISLaSolverGenerator):
            raise ValueError("The ISLaSolverGenerator generator is not compatible with the ParallelEngine")

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

