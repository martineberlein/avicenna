import re
import os
import random
import string
from typing import Union

from fuzzingbook.Grammars import srange

from debugging_framework.oracle import OracleResult
from avicenna.input import Input


def vulnerable_heartbeat(payload, fake_length):
    memory = payload + str(os.urandom(100))  # Simulating extra memory after payload.
    return memory[:fake_length]


def oracle_simple(test_input: Input | str):
    heartbeat_request = str(test_input).split()
    request_length = int(heartbeat_request[1])
    request_payload = heartbeat_request[2]

    response = vulnerable_heartbeat(request_payload, request_length)
    if len(response) > len(request_payload):
        return OracleResult.FAILING
    elif response == request_payload:
        return OracleResult.PASSING
    return OracleResult.UNDEFINED


grammar = {
    "<start>": ["<heartbeat-request>"],
    "<heartbeat-request>": ["\x01 <payload-length> <payload> <padding>"],
    "<payload-length>": ["<one_nine><maybe_digits>"],
    "<one_nine>": srange("123456789"),
    "<maybe_digits>": [
        "",
        "<digits>",
    ],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<digit>": list(string.digits),
    "<payload>": ["<string>"],
    "<padding>": ["<string>"],
    "<string>": ["<char>", "<char><string>"],
    "<char>": list(string.ascii_letters),
}

initial_inputs = [
    "\x01 5 HELLO PADDING",
    "\x01 7 ILoveSE padding",
    "\x01 100 Hello RANDOM",
]


def heartbeat_string_to_hex(s):
    """Convert a string conforming to the heartbeat protocol to its hexadecimal representation."""
    # Separate the type from the rest of the string
    type_byte = s[0]  # This is now a single byte
    remaining_string = s[1:].lstrip()  # Remove leading spaces

    # Use regex to parse the rest of the string
    match = re.match(r"(\d+) ([\w\s]+)", remaining_string)
    if not match:
        raise ValueError("Invalid heartbeat string format")

    # Extract components from the match
    length = int(match.group(1))
    payload_and_padding = match.group(2).replace(" ", "")

    # Check if length corresponds to actual payload
    # if len(payload_and_padding) < length:
    #    raise ValueError("Payload length doesn't match the provided length value")

    # Extract payload based on the length
    payload = payload_and_padding[:length]
    padding = payload_and_padding[length:]

    # Convert each section to hex and concatenate
    type_hex = f"{ord(type_byte):02X}"
    length_bytes = length.to_bytes(2, byteorder="big")
    payload_bytes = payload.encode("utf-8")
    padding_bytes = padding.encode("utf-8")

    hex_representation = " ".join(
        f"{byte:02X}"
        for byte in (
            bytes.fromhex(type_hex) + length_bytes + payload_bytes + padding_bytes
        )
    )
    return hex_representation


def hex_to_heartbeat_string(hex_string):
    """Convert a hexadecimal representation to its heartbeat protocol string format."""

    # Convert the hex string to bytes
    byte_data = bytes.fromhex(hex_string.replace(" ", ""))

    # Extract components from the byte data
    type_byte = byte_data[0]
    length_bytes = byte_data[1:3]
    payload_length = int.from_bytes(length_bytes, byteorder="big")

    # Extract payload and padding based on the length
    payload = byte_data[3 : 3 + payload_length].decode("utf-8", errors="replace")
    padding = byte_data[3 + payload_length :].decode(
        "utf-8", errors="replace"
    )  # Using 'ignore' to handle potential non-UTF8 characters

    # Construct the heartbeat string
    heartbeat_string = f"\\x{type_byte:02X} {payload_length} {payload} {padding}"
    return heartbeat_string


def generate_random_utf8_string(length):
    """Generate a random UTF-8 string of the given length."""
    characters = (
        string.ascii_letters + string.digits + string.punctuation
    )  # + string.whitespace
    return "".join(random.choice(characters) for _ in range(length))


def heartbeat_response(hex_request):
    """Simulate a vulnerable Heartbeat response."""

    # Convert the hex string to bytes
    byte_data = bytes.fromhex(hex_request.replace(" ", ""))

    # Extract the payload length from the request
    length_bytes = byte_data[1:3]
    payload_length = int.from_bytes(length_bytes, byteorder="big")

    # Simulated memory: Actual payload + some extra memory
    # memory = byte_data[3:] + os.urandom(1000)  # 1000 bytes of extra memory for demonstration
    memory = byte_data[3:] + generate_random_utf8_string(1000).encode("utf-8")

    # Retrieve the payload and padding based on the specified length
    response_payload = memory[:payload_length]

    # Construct the official Heartbeat response
    response_type = "02 "
    response = response_type + " ".join(
        f"{byte:02X}" for byte in (length_bytes + response_payload)
    )

    return response


def _test_heartbleed_vulnerability(request_str, response_hex):
    """Test if the Heartbleed bug occurred based on the request and response."""

    # Extract the specified payload length from the request string format
    # Given the format "\x01 8 Hello abc", we extract the number after the space and before the payload.
    specified_payload_length = int(request_str.split()[1])

    # Extract the actual payload from the request string (assuming space-separated format)
    actual_payload = request_str.split()[2][:specified_payload_length].encode("utf-8")

    # Convert the response hex string to bytes
    response_byte_data = bytes.fromhex(response_hex.replace(" ", ""))

    # Extract the payload (and potentially extra data) from the response
    response_payload_and_extra = response_byte_data[
        3 : 3 + specified_payload_length + len(actual_payload)
    ]

    # Compare the response payload to the request payload
    # If they're not the same or if there's extra data, then there's a potential Heartbleed vulnerability
    if response_payload_and_extra != actual_payload:
        return True  # Heartbleed bug might have occurred

    return False


def oracle(test_input: Union[Input, str]) -> OracleResult:
    try:
        heartbeat_request_str = str(test_input)
        hex_request = heartbeat_string_to_hex(heartbeat_request_str)
        response = heartbeat_response(hex_request)
        is_vulnerable = _test_heartbleed_vulnerability(heartbeat_request_str, response)
    except OverflowError:
        return OracleResult.UNDEFINED
    return OracleResult.FAILING if is_vulnerable else OracleResult.PASSING
