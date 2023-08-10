import string
from fuzzingbook.Grammars import srange

from avicenna.input import Input
from avicenna.oracle import OracleResult

import os


def vulnerable_heartbeat(payload, fake_length):
    memory = payload + str(os.urandom(100))  # Simulating extra memory after payload.
    return memory[:fake_length]


def oracle(test_input: Input | str):
    heartbeat_request = str(test_input).split()
    request_length = int(heartbeat_request[1])
    request_payload = heartbeat_request[2]

    response = vulnerable_heartbeat(request_payload, request_length)
    if len(response) > len(request_payload):
        return OracleResult.BUG
    elif response == request_payload:
        return OracleResult.NO_BUG
    return OracleResult.UNDEF


grammar = {
    "<start>": ["<heartbeat-request>"],
    "<heartbeat-request>": ["\x01 <payload-length> <payload> <padding>"],
    "<payload-length>": ["<one_nine><maybe_digits>"],
    "<one_nine>": srange("123456789"),
    "<maybe_digits>": ["", "<digits>", ],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<digit>": list(string.digits),
    "<payload>": ["<string>"],
    "<padding>": ["<string>"],
    "<string>": ["<char>", "<char><string>"],
    "<char>": list(string.ascii_letters),
}

initial_inputs = ["\x01 5 Hello abc", "\x01 7 ILoveSE padding", "\x01 9 Hello RANDOM"]


if __name__ == "__main__":
    for inp in initial_inputs:
        inp_ = Input.from_str(grammar, inp, oracle(inp))
        print(str(inp_).encode(), inp_.oracle)
