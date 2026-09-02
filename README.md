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

Building requires a C++20 compiler and the Python packages `meson`, `meson-python`, `ninja` and `pybind11`. From the root directory of the package:

```shell
pip3 install .
```

For development, `make ffcore` compiles the binary `ffcore...` in place inside the `fastfermion` subdirectory (`make test` also runs the test suite). The package can then be imported by adding the root fastfermion directory to your path, e.g.,

```python
>>> import sys
>>> sys.path.insert(0,"/path/to/fastfermion")
>>> import fastfermion
```

For systems of at most 64 qubits, the meson option `-Dkey_words=1` makes the Pauli string key a single machine word, which is faster (`pip3 install -Csetup-args=-Dkey_words=1 .`).

You could also use the library directly in your C++ project (even though the library was primarily intended to be used in Python). It is header-only, so you can just include the relevant header files from `src/`.
