"""
To be imported by any test file
This sets the path to be able to import fastfermion
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import itertools
import openfermion as of
import numpy as np

import fastfermion as ff
from fastfermion import (
    FermiPolynomial,
    FermiString,
    MajoranaPolynomial,
    MajoranaString,
    PauliPolynomial,
    PauliString
)


STRING_TYPES = {
    PauliPolynomial: PauliString,
    FermiPolynomial: FermiString,
    MajoranaPolynomial: MajoranaString
}

def check_equal(a: PauliPolynomial | FermiPolynomial | MajoranaPolynomial,
                b: of.QubitOperator | of.FermionOperator | of.MajoranaOperator):
    """Checks if a fastfermion polynomial and an OpenFermion polynomial are equal"""
    if isinstance(b, of.FermionOperator):
        b = of.normal_ordered(b)
    assert (isinstance(a, PauliPolynomial) and isinstance(b, of.QubitOperator)) \
        or (isinstance(a, FermiPolynomial) and isinstance(b, of.FermionOperator)) \
        or (isinstance(a, MajoranaPolynomial) and isinstance(b, of.MajoranaOperator))
    a_string_type = STRING_TYPES[type(a)]
    assert len(a.terms) == len(b.terms), f"{a} and {b} don't have the same number of terms"
    for s,sv in b.terms.items():
        # s is of the form
        #   ((0,'X'),(1,'Y')) in the case of PauliOperator
        #   ((3,1),(2,0)) in the case of FermionOperator
        #   (1,3,9) in the case of MajoranaOpeator   
        assert a.terms.get(a_string_type(s)) == sv, f"{a} and {b} differ on the coefficient of {s}"


def check_approx_equal(a: PauliPolynomial | FermiPolynomial | MajoranaPolynomial,
                       b: of.QubitOperator | of.FermionOperator | of.MajoranaOperator,
                       threshold=1e-7):
    """Checks if a fastfermion polynomial and an OpenFermion polynomial are approximately equal"""
    if isinstance(b, of.FermionOperator):
        b = of.normal_ordered(b)
    assert (isinstance(a, PauliPolynomial) and isinstance(b, of.QubitOperator)) \
        or (isinstance(a, FermiPolynomial) and isinstance(b, of.FermionOperator)) \
        or (isinstance(a, MajoranaPolynomial) and isinstance(b, of.MajoranaOperator))
    for s,sv in a.terms.items():
        b -= type(b)(s.indices(),sv)
    for s,sv in b.terms.items():
        assert abs(sv) < threshold, f"Polynomials differ on term {s}, diff is {sv}"


def check_sparse_matrices_exactly_equal(a, b):
    """Check that two SciPy sparse matrices are exactly equal"""
    assert (a-b).count_nonzero() == 0

def check_sparse_matrices_approx_equal(a, b, tol=1e-8):
    """Check that two SciPy sparse matrices are approx equal"""
    assert np.max(np.abs(a.todense() - b.todense())) < tol

def check_paulipolynomial_equal(
        a: ff.PauliPolynomial,
        b: ff.PauliPolynomial
    ):
    """Checks that two PauliPolynomials are approximately equal"""
    assert (a - b).norm('inf') == 0, "Polynomials are not equal"

def check_paulipolynomial_approx_equal(
        a: ff.PauliPolynomial,
        b: ff.PauliPolynomial
    ):
    """Checks that two PauliPolynomials are approximately equal"""
    assert (a - b).norm('inf') <= 1e-14, "Polynomials are not equal"

def check_majoranapolynomial_equal(
        a: ff.MajoranaPolynomial,
        b: ff.MajoranaPolynomial
    ):
    """Checks that two MajoranaPolynomials are approximately equal"""
    assert (a - b).norm('inf') == 0, f"Polynomials {a} and {b} are not equal"

def check_majoranapolynomial_approx_equal(
        a: ff.MajoranaPolynomial,
        b: ff.MajoranaPolynomial
    ):
    """Checks that two MajoranaPolynomials are approximately equal"""
    assert (a - b).norm('inf') <= 1e-14, f"Polynomials {a} and {b} are not equal"

def subsets(iterable):
    """Returns all subsets of iterable"""
    pool = tuple(iterable)
    n = len(pool)
    for mask in itertools.product([False, True], repeat=n):
        yield tuple(pool[i] for i in range(n) if mask[i])

def paulistringstuple(n):
    """Returns iterator on all PauliStrings on n qubits as tuples"""
    for indop in itertools.product(["I","X","Y","Z"],repeat=n):
        yield tuple((ind,op) for ind,op in enumerate(indop) if op != "I")

def fermistringstuple(n):
    """Returns iterator on all FermiStrings on n modes as tuples"""
    for cre in subsets(range(n)):
        for ann in subsets(range(n)):
            yield tuple((u,1) for u in reversed(cre)) + tuple((s,0) for s in reversed(ann))

def majoranastringstuple(n):
    """Returns iterator on all MajoranaStrings as tuples"""
    return subsets(range(n))


def _angles(k, seed):
    """k factors for the gate angles: all 1, or random in [1, 1.1) if a seed is given (no two gates equal)"""
    return np.ones(k) if seed is None else 1 + 0.1 * np.random.default_rng(seed).random(k)

def tfim_circuit(n, dt, seed=None):
    """One Trotter step of the transverse-field Ising ring on n qubits"""
    a = _angles(2 * n, seed)
    return [ff.ROT("ZZ", [i, (i + 1) % n], -2 * dt * a[i]) for i in range(n)] + \
           [ff.ROT("X", [i], -2 * dt * a[n + i]) for i in range(n)]

def heisenberg_circuit(n, dt, seed=None):
    """One Trotter step of the Heisenberg ring on n qubits"""
    a = _angles(3 * n, seed)
    return [ff.ROT(op, [i, (i + 1) % n], 2 * dt * a[3 * i + k]) for i in range(n) for k, op in enumerate(("XX", "YY", "ZZ"))]

def tv_circuit(M, t, V, dt, steps, seed=None):
    """steps Trotter steps of the spinless t-V chain on M modes as Majorana rotations"""
    a = _angles(5 * (M - 1), seed)
    gates = []
    for j in range(M - 1):
        m = 2 * j  # first Majorana operator of mode j
        gates.append(ff.MROT(ff.MajoranaString([m + 1, m + 2]), -t * dt * a[5 * j]))
        gates.append(ff.MROT(ff.MajoranaString([m, m + 3]), t * dt * a[5 * j + 1]))
        if V != 0:
            gates.append(ff.MROT(ff.MajoranaString([m, m + 1]), V / 2 * dt * a[5 * j + 2]))
            gates.append(ff.MROT(ff.MajoranaString([m + 2, m + 3]), V / 2 * dt * a[5 * j + 3]))
            gates.append(ff.MROT(ff.MajoranaString([m, m + 1, m + 2, m + 3]), -V / 2 * dt * a[5 * j + 4]))
    return gates * steps

def poly_terms(p):
    """The terms of a polynomial as a dict string -> complex coefficient"""
    return {str(k): complex(v) for k, v in p.terms.items()}

def assert_poly_close(a, b, tol=1e-12):
    """Checks that two polynomials have the same terms, with coefficients within tol"""
    da, db = poly_terms(a), poly_terms(b)
    assert da.keys() == db.keys(), f"terms differ: {da.keys() ^ db.keys()}"
    for k in da:
        assert abs(da[k] - db[k]) <= tol, f"coefficient of {k}: {da[k]} vs {db[k]}"
