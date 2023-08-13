import unittest
from avicenna_formalizations.heartbeat import *

# Simulate a Heartbleed request
hex_request = "01 00 64 48 65 6C 6C 6F 52 41 4E 44 4F 4D"
print("Received Heartbeat request:", hex_request)

response = heartbeat_response(hex_request)
print("Heartbleed response:", response)

decoded_response = hex_to_heartbeat_string(response)
print("Decoded Response:", decoded_response)


class TestHeartbeatProtocol(unittest.TestCase):
    def test_something(self):
        # hex_representation = "01 00 05 48 45 4C 4C 4F 52 41 4E 44 4F 4D 50 41 44 44 49 4E 47"
        # print(hex_to_heartbeat_string(hex_representation))

        for inp in ["\x01 5 HELLO RANDOMPADDING", "\x01 5 Hello abc"]:
            request = heartbeat_string_to_hex(inp)
            print("Send: ", request)
            print("Received: ", hex_to_heartbeat_string(request))

        for inp in initial_inputs:
            inp_ = Input.from_str(grammar, inp, oracle(inp))
            print(str(inp_).encode('utf-8'), inp_.oracle)

if __name__ == '__main__':
    unittest.main()
