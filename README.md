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
* Heisenberg evolution: Propagate polynomial through a sequence of unitaries/gates with truncation by degree, coefficient magnitude or number of terms, on one or several threads (OpenMP) or on a GPU (CUDA)
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

Building requires a C++20 compiler (`pip` installs the build dependencies `meson`, `meson-python`, `ninja` and `pybind11` itself). From the root directory of the package:

```shell
pip3 install .
```

For development, `make ffcore` compiles the binary `ffcore...` in place inside the `fastfermion` subdirectory (`make test` also runs the test suite). The package can then be imported by adding the root fastfermion directory to your path, e.g.,

```python
>>> import sys
>>> sys.path.insert(0,"/path/to/fastfermion")
>>> import fastfermion
```

OpenMP is used when the compiler supports it: `propagate(circuit, observable, n_threads=8)` runs on 8 threads. For systems of at most 64 qubits, the meson option `-Dkey_words=1` makes the Pauli string key a single machine word (and halves the Fermi and Majorana keys), which is faster (`pip3 install -Csetup-args=-Dkey_words=1 .`).

#### GPU backend

With the CUDA toolkit installed (tested with CUDA 12.8), the GPU backend is built automatically (`-Dgpu=enabled` to require it, `-Dgpu=disabled` to skip it; or `make ffcore MESON_ARGS=-Dgpu=enabled`), and `propagate(circuit, observable, maxdegree=6, parallel="gpu")` propagates on the device: at most 128 qubits or modes, with the degree cutoff and the coefficient threshold, on an observable with real coefficients. The meson option `gpu_arch` selects the target architecture (`native` by default, e.g., `-Dgpu_arch=sm_80` when building on a machine without a GPU).

You could also use the library directly in your C++ project (even though the library was primarily intended to be used in Python). It is header-only, so you can just include the relevant header files from `src/`.
