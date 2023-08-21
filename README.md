[![Python Version](https://img.shields.io/pypi/pyversions/avicenna)](https://pypi.org/project/avicenna/)
[![GitHub release](https://img.shields.io/github/v/release/martineberlein/avicenna)](https://github.com/martineberlein/avicenna/releases)
[![PyPI](https://img.shields.io/pypi/v/avicenna)](https://pypi.org/project/avicenna/)
[![Tests](https://github.com/martineberlein/avicenna/actions/workflows/test_avicenna.yml/badge.svg)](https://github.com/martineberlein/avicenna/actions/workflows/test_avicenna.yml)
[![License](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
&nbsp;

# Avicenna

This repo contains the code to execute, develop and test our debugging prototype **Avicenna**.

**Avicenna** is a debugging tool designed to automatically determine the causes and conditions of program failures.
It leverages both generative and predictive models to satisfy constraints over grammar elements and detect relations of input elements.
Our tool uses the [ISLa](https://github.com/rindPHI/isla) specification language to express complex failure circumstances as predicates over input elements.
**Avicenna** learns input properties that are common across failing inputs and employs a feedback loop to refine the current debugging diagnoses by systematic experimentation.
The result is crisp and precise diagnoses that closely match those determined by human experts, offering a significant advancement in the realm of automated debugging.


## Quickstart

To demonstrate Avicenna's capabilities, we'll walk through a simple example using a program we call **The Calculator**. 

The Calculator is a Python program designed to evaluate various mathematical expressions, including arithmetic equations and trigonometric functions:

```python
import math

def arith_eval(inp) -> float:
    return eval(
        str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
    )
```

To identify bugs in The Calculator, we'll implement an oracle function. This function tests inputs and categorizes them as producing expected behavior (`OracleResult.NO_BUG`) or a bug (`OracleResult.BUG`):

```python 
from avicenna.oracle import OracleResult
from avicenna.input import Input

def oracle(inp: str | Input) -> OracleResult:
    try:
        arith_eval(inp)
        return OracleResult.NO_BUG
    except ValueError:
        return OracleResult.BUG

    return OracleResult.NO_BUG
``` 

To see the oracle function in action, we'll test a few sample inputs:

```python
initial_inputs = ['sqrt(1)', 'cos(912)', 'tan(4)', 'sqrt(-3)']

for inp in initial_inputs:
    print(inp.ljust(30), oracle(inp))
```

Executing this code provides the following output:

```
sqrt(1)                        NO_BUG
cos(912)                       NO_BUG
tan(4)                         NO_BUG
sqrt(-3)                       BUG
```

As we can see, the input `sqrt(-3)` triggers a bug in The Calculator. 

In the following steps, we'll leverage Avicenna to pinpoint the root cause of this bug.
We'll employ Avicenna's capabilities to identify the root cause of the bug and provide potential fixes.
This will involve defining the input structure for our calculator and initializing Avicenna with the appropriate grammar, sample inputs, and oracle function.

To define the input structure, we utilize a grammar:

```python
grammar = {
    "<start>": ["<arith_expr>"],
    "<arith_expr>": ["<function>(<number>)"],
    "<function>": ["sqrt", "sin", "cos", "tan"],
    "<number>": ["<maybe_minus><onenine><maybe_digits><maybe_frac>"],
    "<maybe_minus>": ["", "-"],
    "<onenine>": [str(num) for num in range(1, 10)],
    "<digit>": [str(num) for num in range(0, 10)],
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<maybe_frac>": ["", ".<digits>"],
}
```
The grammar represents the structure of valid inputs that can be given to the Calculator.
For instance, it recognizes mathematical expressions involving functions (*sqrt, sin, cos, tan*) applied to numbers, which can be positive or negative.

Now, we're ready to initiate Avicenna with this grammar, the sample inputs (Note that at least one failure-inducing sample input is required), and the oracle function:

```python
from avicenna import Avicenna

avicenna = Avicenna(
    grammar=grammar,
    initial_inputs=initial_inputs,
    oracle=oracle
)

diagnosis = avicenna.explain()
```
In the code above, we've created an instance of the Avicenna class and started the debugging process by invoking the `explain()` method.
Avicenna will utilize its feedback loop to systematically probe and test the Calculator program, identify the root cause of the bug on the analysis of the bug's behavior.

```python
from isla.language import ISLaUnparser

print(ISLaUnparser(diagnosis[0]).unparse())
```
This output is a symbolic representation -- an ISLa Constraint -- of the root cause of the failure detected by Avicenna in the Calculator program. Here's a breakdown of what it means:

```
(forall <number> elem in start:
        (<= (str.to.int elem) (str.to.int "-1")) and
  exists <function> elem_0 in start:
        (= elem_0 "sqrt"))
```

This output, expressed in first-order logic, is saying:

- For all numbers (elements of type `<number>` in the grammar), if the integer representation of the number is less than or equal to -1 (`<= (str.to.int elem) (str.to.int "-1")`), and
- There exists a function (an element of type `<function>` in the grammar) that equals to "sqrt" (`= elem_0 "sqrt"`),

then a bug is likely to occur.

In plain English, the output is indicating that the failure in our Calculator program occurs when trying to take the square root (`sqrt`) of a negative number (a number less than or equal to -1). 

This is consistent with our expectations, since the square root of a negative number is not defined in the realm of real numbers. Consequently, Python's `math.sqrt()` function, which we've used in our Calculator program, throws a `ValueError` when given a negative number as input.

With this information, we can address the issue in our Calculator program to prevent crashes when dealing with such inputs. We might decide to handle such errors gracefully or implement support for complex numbers, depending on the requirements of our program.

Remember, these results are generated based on the information provided to Avicenna, such as the grammar and the oracle function, as well as the results of Avicenna's systematic testing of the Calculator program. So the more accurate and comprehensive these inputs are, the more helpful Avicenna's outputs will be.

### More Examples:

If you want to explore more of how Avicenna works and how it can, for instance, be used to generate new and unseen behavior-triggering inputs, make sure to have a look at the jupyter notebooks in the [notebooks](./notebooks) folder:
- **[Calculator.ipynb](./notebooks/calculator.ipynb)**: A quick and more detailed tutorial on how to setup up Avicenna. It also showcases how to use the ISLa-Solver to generate new inputs from the final diagnosis.
- **[Heartbleed.ipynb](./notebooks/heartbleed.ipynb)**: This notebook demonstrates the capabilities of Avicenna on the infamous [Heartbleed Bug](https://heartbleed.com).

## Install, Development, Testing

### Install
If all external dependencies are available, a simple pip install avicenna suffices.
We recommend installing Avicenna inside a virtual environment (virtualenv):

```
python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install avicenna
```

### Development and Testing

For development and testing, we recommend using Avicenna inside a virtual environment (virtualenv).
By doing the following steps in a standard shell (bash), one can run the Avicenna tests:

```
git clone https://github.com/martineberlein/avicenna.git
cd avicenna/

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip

# Run tests
pip install -e .[dev]
python3 -m pytest
```

### Docker

For users who want to test Avicenna without managing external dependencies like Python, we offer a Docker container setup. Docker allows you to package and run applications in lightweight containers. Our Docker container already includes all necessary dependencies for Avicenna.

**Step 1:** Build the Docker image. Ensure you are in the directory containing the Dockerfile and execute:

```shell
docker build -t avicenna_image .
```

**Step 2:** Run a Docker container using the image you just built:

```shell
docker run -it --name avicenna avicenna_image
```

After executing the above command, you'll be inside the Docker container's shell.

**Step 3:** Update the Avicenna repository:

```shell
cd avicenna/
git pull
```

**Step 4:** Install Avicenna in editable mode and execute its tests:

```shell
pip install -e .[dev]
python3.10 -m pytest ./tests
```
