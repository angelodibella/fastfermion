[![PyPi](https://img.shields.io/pypi/v/fastfermion.svg)](https://pypi.python.org/pypi/fastfermion/)
[![Docs](https://img.shields.io/badge/docs-grey)](https://www.fastfermion.com/)

**fastfermion** is a Python package written in C++ for the efficient manipulation of polynomials in Pauli, Fermi and Majorana operators.

<p align="center">
<img alt="Computing the Jordan-Wigner transform of a CrO molecule Hamiltonian with > 10^5 terms" src="assets/jwperf.svg" style="height: 120px;" /><br />
<i>
Computing the Jordan-Wigner transform of a CrO molecule Hamiltonian with > 10<sup>5</sup> terms</i>
</p>

**Features**

* Algebraic manipulation of polynomials in Pauli operators, Fermionic creation/annihilation operators, and Majorana operators.
* Fermionic and Majorana operators are automatically put in normal ordered form
* Conversion between Pauli, Fermi, and Majorana representations (Jordan-Wigner and reverse Jordan-Wigner)
* Sparse matrix representations
* Heisenberg evolution: Propagate polynomial through a sequence of unitaries/gates with possible truncation by degree
* Interface with OpenFermion and Cirq
* Up to 200x faster than OpenFermion
* More to come ...

## Installation

fastfermion is available on PyPI:

```shell
pip3 install fastfermion
```

Then you should be able to 

```python
import fastfermion
```

from Python. See the `examples` folder to get started, or check out [this tour of fastfermion](https://www.fastfermion.com/tour/).

### Building from source

Requires a C++20 compiler, Python >= 3.10, and [meson](https://mesonbuild.com/):

```shell
pip install meson meson-python pybind11
pip install --no-build-isolation -e .
```

Or, to compile without installing:

```shell
make build          # runs meson setup + compile
make test           # runs the test suite (needs pytest, scipy)
```

Override the Python interpreter with `make PYTHON=/path/to/python build`.

The C++ library is header-only (`src/*.h`), so it can also be included directly in a C++ project.
