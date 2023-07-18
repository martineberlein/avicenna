# avicenna

This repo contains the code to execute, develop and test our debugging prototype **Avicenna**.

### Example:

```python
from avicenna import Avicenna
import math

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

def prop(inp) -> bool:
    try:
        eval(
            str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
        )   
        return False
    except ValueError:
        return True

initial_inputs = ["cos(10)", "sqrt(28367)", "tan(-12)", "sqrt(-900)"]

avicenna = Avicenna(
    grammar=grammar,
    initial_inputs=initial_inputs,
    evaluation_function=prop,
)

avicenna.execute()
```

The output wil be something like this:

```
exists <payload> container1 in start:
  exists <length> length_field in start:
    (< (str.len container1) (str.to.int length_field))
```

## Install, Development, Testing, Build

### Install
If all external dependencies are available, a simple pip install avicenna suffices.
We recommend installing Avicenna inside a virtual environment (virtualenv):

```
python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install avicenna
```

Now, the avicenna command should be available on the command line within the virtual environment.

### Development and Testing

For development, we recommend using Avicenna inside a virtual environment (virtualenv).
By thing the following steps in a standard shell (bash), one can run the Avicenna tests:

```
git clone https://github.com/martineberlein/semantic-debugging.git
cd semantic-debuging/

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip

# Run tests
pip install -e .[dev]
python3 -m pytest
```

### Build

EvoGFuzz is build locally as follows:

```
git clone https://github.com/martineberlein/semantic-debugging.git
cd semantic-debuging/

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade build
python3 -m build
```
