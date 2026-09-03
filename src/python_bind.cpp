/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#include "pauli/algebra.h"
#include "fermi_algebra.h"
#include "majorana/algebra.h"
#include "pauli/sparse.h"
#include "fermi_sparse.h"
#include "majorana/sparse.h"
#include "transforms.h"
#include "fockstate.h"
#include "qubitproductstate.h"
#include "pauli/gates.h"
#include "pauli/propagate.h"
#include "majorana/gates.h"
#include "majorana/propagate.h"
#include "gen.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

namespace py = pybind11;

// To cast ankerl::unordered_dense::map as a python dict
template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct pybind11::detail::type_caster<ankerl::unordered_dense::map<Key, Value, Hash, Equal, Alloc>>
    : pybind11::detail::map_caster<ankerl::unordered_dense::map<Key, Value, Hash, Equal, Alloc>, Key, Value> {};


namespace fastfermion {

// To cast CSCMatrix to SciPy CSCMatrix
auto CSCMatrix_to_scipy(const CSCMatrix<ff_complex>& sm) {
    py::object csc_matrix_type = py::module_::import("scipy").attr("sparse").attr("csc_matrix");
    return csc_matrix_type(std::make_tuple(std::move(sm.data), std::move(sm.indices), std::move(sm.indptr)), std::make_tuple(sm.shape[0], sm.shape[1])).release();
};

template <typename PolyT, typename StringT, typename MonT>
void add_poly_basic_ops(py::class_<PolyT>& pyPolyClass) {

    pyPolyClass.def("extent", &PolyT::extent);
    pyPolyClass.def("compress", &PolyT::compress, py::arg("threshold")=1e-8);
    pyPolyClass.def("permute", &PolyT::permute);
    pyPolyClass.def("truncate", &PolyT::truncate);
    pyPolyClass.def("commutes", py::overload_cast<const ff_complex&>(&PolyT::commutes, py::const_));
    pyPolyClass.def("commutes", py::overload_cast<const StringT&>(&PolyT::commutes, py::const_));
    pyPolyClass.def("commutes", py::overload_cast<const MonT&>(&PolyT::commutes, py::const_));
    pyPolyClass.def("commutes", py::overload_cast<const PolyT&>(&PolyT::commutes, py::const_));

    pyPolyClass.def("commutator", py::overload_cast<const ff_complex&>(&PolyT::commutator, py::const_));
    pyPolyClass.def("commutator", py::overload_cast<const StringT&>(&PolyT::commutator, py::const_));
    pyPolyClass.def("commutator", py::overload_cast<const MonT&>(&PolyT::commutator, py::const_));
    pyPolyClass.def("commutator", py::overload_cast<const PolyT&>(&PolyT::commutator, py::const_));

    pyPolyClass.def_readonly("terms", &PolyT::terms);
    pyPolyClass.def("__len__", [](const PolyT& a) { return a.terms.size(); });
    pyPolyClass.def("__str__", &PolyT::to_compact_string);
    pyPolyClass.def("__repr__", &PolyT::to_compact_string);
    pyPolyClass.def("norm", [](const PolyT& p, const std::variant<int,std::string>& v) {
        // v can take the values 0, 1, 2, or "inf"
        // That's why we use variant
        if(v.index() == 0) {
            return p.norm(std::get<int>(v));
        } else {
            if(std::get<std::string>(v) == "inf") {
                return p.norm_inf();
            } else {
                throw_error("Invalid argument");
            }
        }
    }, py::arg("v")=1);
    
    // Operator overloading
    // see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#operator-overloading
    
    // In-place operations

    pyPolyClass.def(py::self += ff_complex());
    pyPolyClass.def(py::self += StringT());
    pyPolyClass.def(py::self += py::self);

    pyPolyClass.def(py::self -= ff_complex());
    pyPolyClass.def(py::self -= StringT());
    pyPolyClass.def(py::self -= py::self);

    pyPolyClass.def(py::self *= ff_complex());
    pyPolyClass.def(py::self *= StringT());
    pyPolyClass.def(py::self *= py::self);

    pyPolyClass.def(py::self /= ff_complex());

    pyPolyClass.def(- py::self);
    
    // Binary operations

    pyPolyClass.def(py::self + ff_complex());
    pyPolyClass.def(py::self + StringT());
    pyPolyClass.def(py::self + py::self);
    pyPolyClass.def(py::self - ff_complex());
    pyPolyClass.def(py::self - StringT());
    pyPolyClass.def(py::self - py::self);
    pyPolyClass.def(py::self * ff_complex());
    pyPolyClass.def(py::self * StringT());
    pyPolyClass.def(py::self * py::self);
    pyPolyClass.def(py::self / ff_complex());
    pyPolyClass.def(ff_complex() + py::self);
    pyPolyClass.def(ff_complex() - py::self);
    pyPolyClass.def(ff_complex() * py::self);
    pyPolyClass.def(py::self == py::self);
    pyPolyClass.def(py::self != py::self);

}

template<typename PauliOpT>
void add_pauli_sparse(py::class_<PauliOpT>& pyPauliOpClass) {
    pyPauliOpClass.def("sparse",[](const PauliOpT& a, const std::optional<int>& n, const std::optional<int>& nup) {
        int _n = n.has_value() ? n.value() : a.extent();
        if(!nup.has_value()) {
            return CSCMatrix_to_scipy(a.sparse(_n));
        } else {
            return CSCMatrix_to_scipy(a.sparse(_n,nup.value()));
        }
    }, py::arg("n") = py::none(), py::arg("nup") = py::none());
}

template<typename FermiOpT>
void add_fermi_sparse(py::class_<FermiOpT>& pyFermiOpClass) {
    pyFermiOpClass.def("sparse",[](const FermiOpT& a, const std::optional<int>& n, const std::optional<int>& nocc) {
        int _n = n.has_value() ? n.value() : a.extent();
        if(!nocc.has_value()) {
            return CSCMatrix_to_scipy(a.sparse(_n));
        } else {
            return CSCMatrix_to_scipy(a.sparse(_n,nocc.value()));
        }
    }, py::arg("n") = py::none(), py::arg("nocc") = py::none());
}

template<typename MajoranaOpT>
void add_majorana_sparse(py::class_<MajoranaOpT>& pyMajoranaOpClass) {
    pyMajoranaOpClass.def("sparse",[](const MajoranaOpT& a, const std::optional<int>& n) {
        if(n.has_value()) {
            return CSCMatrix_to_scipy(a.sparse(n.value()));
        } else {
            return CSCMatrix_to_scipy(a.sparse());
        }
    }, py::arg("n") = py::none());
}

void add_transforms(py::module_& m) {

    // Majorana to Pauli and Fermi to Pauli
    m.def("jw", py::overload_cast<const MajoranaString&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("jw", py::overload_cast<const MajoranaPolynomial&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("jw", py::overload_cast<const FermiString&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("jw", py::overload_cast<const FermiPolynomial&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");

    // Pauli to Fermi
    m.def("rjw", py::overload_cast<const PauliString&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");
    m.def("rjw", py::overload_cast<const PauliPolynomial&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");

    // To Pauli
    m.def("topauli", py::overload_cast<const MajoranaString&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("topauli", py::overload_cast<const MajoranaPolynomial&>(&jordan_wigner), "Convert a Majorana polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("topauli", py::overload_cast<const FermiString&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");
    m.def("topauli", py::overload_cast<const FermiPolynomial&>(&jordan_wigner), "Convert a Fermi polynomial to a Pauli polynomial via the Jordan-Wigner mapping");


    // To Fermi
    m.def("tofermi", py::overload_cast<const PauliString&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");
    m.def("tofermi", py::overload_cast<const PauliPolynomial&>(&reverse_jordan_wigner), "Convert a Pauli polynomial to a Fermi polynomial via the reverse Jordan-Wigner mapping");
    m.def("tofermi", py::overload_cast<const MajoranaString&>(&majorana_to_fermi), "Convert a Majorana polynomial a Fermi polynomial");
    m.def("tofermi", py::overload_cast<const MajoranaPolynomial&>(&majorana_to_fermi), "Convert a Majorana polynomial a Fermi polynomial");


    // To Majorana
    m.def("tomajorana", py::overload_cast<const FermiString&>(&fermi_to_majorana), "Convert a Fermi polynomial to Majorana polynomial");
    m.def("tomajorana", py::overload_cast<const FermiPolynomial&>(&fermi_to_majorana), "Convert a Fermi polynomial to Majorana polynomial");
    m.def("tomajorana", [](const PauliString& a) {
        return MajoranaPolynomial(pauli_to_majorana(a));
    }, "Convert a Pauli polynomial to a Majorana polynomial");
    m.def("tomajorana", py::overload_cast<const PauliPolynomial&>(&pauli_to_majorana), "Convert a Pauli polynomial to a Majorana polynomial");

}

void add_states(py::module_& m) {

    py::class_<FockState>(m, "FockState")
        .def(py::init<>())
        .def(py::init<const std::uint64_t&>())
        .def(py::init<const std::vector<int>&>(), py::arg("occ")=std::vector<int>{})
        .def("__call__", py::overload_cast<const FermiString&>(&FockState::operator(), py::const_))
        .def("__call__", py::overload_cast<const FermiPolynomial&>(&FockState::operator(), py::const_))
        .def("__call__", py::overload_cast<const MajoranaString&>(&FockState::operator(), py::const_))
        .def("__call__", py::overload_cast<const MajoranaPolynomial&>(&FockState::operator(), py::const_))
        .def("occ", &FockState::occ)
        .def("vec", [](const FockState& state, const int& n, const std::optional<int>& nocc) {
            if(nocc.has_value()) {
                return state.vec(n, nocc.value());
            } else {
                return state.vec(n);
            }
        }, py::arg("n"), py::arg("nocc") = py::none());

    py::class_<QubitProductState>(m, "QubitProductState")
        .def(py::init<>())
        .def(py::init<const std::uint64_t&>())
        .def(py::init<const std::vector<int>&>(), py::arg("up")=std::vector<int>{})
        .def("__call__", py::overload_cast<const PauliString&>(&QubitProductState::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&QubitProductState::operator(), py::const_))
        .def("up", &QubitProductState::up)
        .def("vec", [](const QubitProductState& state, const int& n, const std::optional<int>& nup) {
            if(nup.has_value()) {
                return state.vec(n, nup.value());
            } else {
                return state.vec(n);
            }
        }, py::arg("n"), py::arg("nocc") = py::none());

}

void add_pauli_gates(py::module_& m) {
    py::class_<pauli_gates::H>(m, "H")
        .def(py::init<const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::H::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::H::operator(), py::const_))
        .def("__repr__", &pauli_gates::H::to_string)
        .def("__str__", &pauli_gates::H::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::H& gate) { return std::vector<int>{gate.i}; })
        .def("aspoly", &pauli_gates::H::aspoly)
        .doc() = R"DOC(
        Hadamard unitary on qubit p:
            U = (X_p + Z_p) / sqrt(2)

        Example:
        >>> from fastfermion import H
        >>> U = H(0)
        )DOC";

    py::class_<pauli_gates::S>(m, "S")
        .def(py::init<const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::S::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::S::operator(), py::const_))
        .def("__repr__", &pauli_gates::S::to_string)
        .def("__str__", &pauli_gates::S::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::S& gate) { return std::vector<int>{gate.i}; })
        .def("aspoly", &pauli_gates::S::aspoly)
        .doc() = R"DOC(
        S unitary on qubit p:
            U = ((1+1j) + (1-1j) Z_p) / 2

        Example:
        >>> from fastfermion import S
        >>> U = S(0)
        )DOC";

    py::class_<pauli_gates::CNOT>(m, "CNOT")
        .def(py::init<const int&, const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::CNOT::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::CNOT::operator(), py::const_))
        .def("__repr__", &pauli_gates::CNOT::to_string)
        .def("__str__", &pauli_gates::CNOT::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::CNOT& gate) { return std::vector<int>{gate.i,gate.j}; })
        .def("aspoly", &pauli_gates::CNOT::aspoly)
        .doc() = R"DOC(
        CNOT unitary on qubits p,q:
            U = (1 + Z_p + X_q - Z_p X_q)/2

        Example:
        >>> from fastfermion import CNOT
        >>> U = CNOT(0,1)
        )DOC";

    py::class_<pauli_gates::SWAP>(m, "SWAP")
        .def(py::init<const int&, const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::SWAP::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::SWAP::operator(), py::const_))
        .def("__repr__", &pauli_gates::SWAP::to_string)
        .def("__str__", &pauli_gates::SWAP::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::SWAP& gate) { return std::vector<int>{gate.i,gate.j}; })
        .def("aspoly", &pauli_gates::SWAP::aspoly)
        .doc() = R"DOC(
        SWAP unitary on qubits p,q:
            U = (1 + X_p X_q + Y_p Y_q + Z_p Z_q)/2

        Example:
        >>> from fastfermion import SWAP
        >>> U = SWAP(0,1)
        )DOC";

    py::class_<pauli_gates::CZ>(m, "CZ")
        .def(py::init<const int&, const int&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::CZ::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::CZ::operator(), py::const_))
        .def("__repr__", &pauli_gates::CZ::to_string)
        .def("__str__", &pauli_gates::CZ::to_string)
        .def_property_readonly("qubits", [](const pauli_gates::CZ& gate) { return std::vector<int>{gate.i,gate.j}; })
        .def("aspoly", &pauli_gates::CZ::aspoly)
        .doc() = R"DOC(
        SWAP unitary on qubits p,q:
            U = (1 + Z_p + Z_q + Z_p Z_q)/2

        Example:
        >>> from fastfermion import CZ
        >>> U = CZ(0,1)
        )DOC";

    py::class_<pauli_gates::ROT>(m, "ROT")
        .def(py::init<const std::string&, const std::vector<int>&, const ff_float&>())
        .def(py::init<const PauliString&, const ff_float&>())
        .def(py::init<const std::string&, const ff_float&>())
        .def("__call__", py::overload_cast<const PauliString&>(&pauli_gates::ROT::operator(), py::const_))
        .def("__call__", py::overload_cast<const PauliPolynomial&>(&pauli_gates::ROT::operator(), py::const_))
        .def_property_readonly("qubits", [](const pauli_gates::ROT& gate) { return gate.ps.support_set(); })
        .def_readonly("axis", &pauli_gates::ROT::ps)
        .def_readonly("theta", &pauli_gates::ROT::theta)
        .def("aspoly", &pauli_gates::ROT::aspoly)
        .def("__repr__", &pauli_gates::ROT::to_string)
        .def("__str__", &pauli_gates::ROT::to_string)
        .doc() = R"DOC(
        Represents a Pauli rotation U = e^{-i P theta/2}
        where P is a Pauli string and theta is real number

        Example:
        >>> from fastfermion import ROT
        >>> U = ROT("XX",(0,1),0.75)
        )DOC";
}

// propagate() releases the GIL, so the warning reacquires it
void warn_if_no_openmp(int n_threads) {
#ifndef FF_OPENMP
    if(n_threads > 1) {
        py::gil_scoped_acquire gil;
        if(PyErr_WarnEx(PyExc_RuntimeWarning, "n_threads > 1 ignored: fastfermion was built without OpenMP", 1) < 0) {
            throw py::error_already_set();
        }
    }
#endif
}

void add_propagation_common(py::module_& m) {

    m.attr("has_openmp") =
#ifdef FF_OPENMP
        true;
#else
        false;
#endif

    m.def("trunc_stats", []() {
        const TruncStats& stats = trunc_stats();
        py::dict d;
        d["cert_tau"] = stats.cert_tau;
        d["n_tau_events"] = stats.n_tau_events;
        d["n_w_events"] = stats.n_w_events;
        d["peak_terms"] = stats.peak_terms;
        return d;
    }, R"DOC(
        Statistics of the last call to propagate (Pauli or Majorana) in the calling thread:
        * cert_tau: bound on the norm of the difference between the result and the result of the same
          propagation without mincoeff (the sum over threshold events of the norm of the discarded terms);
          the other rules are not certified
        * n_tau_events: number of times the mincoeff rule fired
        * n_w_events: number of times a maxdegree rule with maxdegree_period > 1 fired
        * peak_terms: largest number of terms held before a truncation
        )DOC"
    );

}

void add_pauli_propagation(py::module_& m) {

    m.def("propagate",
        [](
            const pauli_gates::Circuit& circuit,
            const std::variant<PauliString, PauliPolynomial>& observable,
            const std::optional<int>& maxdegree,
            const std::optional<ff_float>& mincoeff,
            const std::optional<int>& topk,
            const std::optional<int>& max_xweight,
            int maxdegree_period,
            int mincoeff_period,
            int xweight_period,
            bool batched,
            int n_threads,
            const std::string& parallel
        ) {
            pauli_gates::PauliTruncation truncation;
            warn_if_no_openmp(n_threads);
            truncation.maxdegree = maxdegree.value_or(INT_MAX);
            truncation.mincoeff = mincoeff.value_or(0);
            truncation.topk = topk.value_or(0);
            truncation.max_xweight = max_xweight.value_or(-1);
            truncation.maxdegree_period = maxdegree_period;
            truncation.mincoeff_period = mincoeff_period;
            truncation.xweight_period = xweight_period;
            const PauliPolynomial obs = observable.index() == 0 ? PauliPolynomial(std::get<0>(observable)) : std::get<1>(observable);
            return pauli_gates::propagate(circuit, obs, truncation, batched, n_threads, parallel);
        },
        py::arg("circuit"), py::arg("observable"), py::arg("maxdegree") = py::none(), py::arg("mincoeff") = py::none(),
        py::arg("topk") = py::none(), py::arg("max_xweight") = py::none(), py::arg("maxdegree_period") = 1,
        py::arg("mincoeff_period") = 1, py::arg("xweight_period") = 1, py::arg("batched") = false,
        py::arg("n_threads") = 1, py::arg("parallel") = "auto",
        py::call_guard<py::gil_scoped_release>(),
        R"DOC(
        Backpropagates a polynomial through a circuit.

        Truncation rules, all off by default:
        * maxdegree: discards the terms of degree larger than maxdegree. By default the truncation
          happens as soon as a term is created by a non-Clifford gate (i.e., a ROT gate), so the output
          has degree larger than maxdegree only through Clifford gates or the initial observable. With
          maxdegree_period=p > 1 the terms are instead only discarded after every p-th ROT gate.
        * mincoeff: discards the terms of magnitude at most mincoeff after every mincoeff_period-th ROT gate.
        * topk: keeps only the topk terms of largest magnitude, after every ROT gate.
        * max_xweight: discards the terms with more than max_xweight X or Y factors, as they are created
          by default or after every xweight_period-th ROT gate if xweight_period > 1.

        With batched=True, consecutive commuting ROT gates are applied together and the rules fire once
        per such batch (a rule with period p fires after every batch containing a p-th gate), which
        changes the result under mincoeff or topk. trunc_stats() returns statistics of the last call,
        including a bound on the error due to mincoeff.

        With n_threads > 1 the propagation runs on that many OpenMP threads (a build without OpenMP
        ignores it), the terms being partitioned between the threads (parallel="sharded", the default
        when n_threads > 1; parallel="serial" forces one thread). The results do not depend on the
        number of threads, up to rounding (which may change the terms topk keeps in case of ties).

        Examples:
        >>> from fastfermion import H, CNOT, propagate
        >>> circuit = [H(0),CNOT(0,1)]
        >>> observable = PauliString("XZ")
        >>> result = propagate(circuit,observable,maxdegree=3,mincoeff=1e-8)
        )DOC"
    );

}

void add_majorana_propagation(py::module_& m) {

    py::class_<majorana_gates::MROT>(m, "MROT")
        .def(py::init<const std::vector<int>&, const ff_complex&, const ff_float&>())
        .def(py::init<const MajoranaString&, const ff_float&>())
        .def(py::init<const MajoranaString&, const ff_complex&, const ff_float&>())
        .def("__call__", py::overload_cast<const MajoranaString&>(&majorana_gates::MROT::operator(), py::const_))
        .def("__call__", py::overload_cast<const MajoranaPolynomial&>(&majorana_gates::MROT::operator(), py::const_))
        .def_property_readonly("qubits", [](const majorana_gates::MROT& gate) { return gate.ms.support_set(); })
        .def_readonly("axis", &majorana_gates::MROT::ms)
        .def_readonly("theta", &majorana_gates::MROT::theta)
        .def("aspoly", &majorana_gates::MROT::aspoly)
        .def("__repr__", &majorana_gates::MROT::to_string)
        .def("__str__", &majorana_gates::MROT::to_string)
        .doc() = R"DOC(
        Represents a Majorana rotation U = e^{-i theta/2 M}
        where M is a Hermitian Majorana monomial and theta is a real number.
        The Majorana monomial is supplied as a pair P,c where P is
        a MajoranaString and c is a complex number, so that
            MROT(P,c,theta)
        represents the unitary e^{-i theta/2 (c*P)}.
        If c*P is not Hermitian, an error is raised

        Example:
        >>> from fastfermion import MROT, MajoranaString
        >>> U = MROT(MajoranaString([0,1]),1j,1.0)
        >>> print(U)
        MROT(m0 m1,1.000000) = e^{0.500000 m0 m1}
        >>> V = MROT(MajoranaString([0,2]),1,2.0)
        RuntimeError: Supplied Majorana monomial 1*(m0 m2) is not Hermitian
        )DOC";

    m.def("propagate",
        [](
            const majorana_gates::MajoranaCircuit& circuit,
            const std::variant<MajoranaString,MajoranaPolynomial>& observable,
            const std::optional<int>& maxdegree,
            const std::optional<ff_float>& mincoeff,
            const std::optional<int>& topk,
            const std::optional<int>& max_unpaired,
            int maxdegree_period,
            int mincoeff_period,
            int unpaired_period,
            bool batched,
            int n_threads,
            const std::string& parallel
        ) {
            majorana_gates::MajoranaTruncation truncation;
            warn_if_no_openmp(n_threads);
            truncation.maxdegree = maxdegree.value_or(INT_MAX);
            truncation.mincoeff = mincoeff.value_or(0);
            truncation.topk = topk.value_or(0);
            truncation.max_unpaired = max_unpaired.value_or(-1);
            truncation.maxdegree_period = maxdegree_period;
            truncation.mincoeff_period = mincoeff_period;
            truncation.unpaired_period = unpaired_period;
            const MajoranaPolynomial obs = observable.index() == 0 ? MajoranaPolynomial(std::get<0>(observable)) : std::get<1>(observable);
            return majorana_gates::propagate(circuit, obs, truncation, batched, n_threads, parallel);
        },
        py::arg("circuit"), py::arg("observable"), py::arg("maxdegree") = py::none(), py::arg("mincoeff") = py::none(),
        py::arg("topk") = py::none(), py::arg("max_unpaired") = py::none(), py::arg("maxdegree_period") = 1,
        py::arg("mincoeff_period") = 1, py::arg("unpaired_period") = 1, py::arg("batched") = false,
        py::arg("n_threads") = 1, py::arg("parallel") = "auto",
        py::call_guard<py::gil_scoped_release>(),
        R"DOC(
        Backpropagates a Majorana polynomial through a Majorana circuit.

        The truncation rules maxdegree, mincoeff and topk, their periods and batched are as in the
        Pauli propagate. max_unpaired discards the terms with more than max_unpaired unpaired modes
        (modes with exactly one of their two Majorana operators in the term), as they are created by
        default or after every unpaired_period-th gate if unpaired_period > 1; such terms have zero
        expectation in every Fock state. n_threads and parallel are as in the Pauli propagate.

        Examples:
        >>> from fastfermion import MROT, MajoranaString, propagate
        >>> circuit = [MROT(MajoranaString([0,1]),1j,1.0)]
        >>> observable = MajoranaString([0,2])
        >>> result = propagate(circuit,observable,maxdegree=3)
        )DOC"
    );

}


void add_gen(py::module_& m) {

    m.def("paulis",&paulis,R"DOC(
            Returns generators of Pauli algebra

            Example:
                >>> from fastfermion import paulis
                >>> X,Y,Z = paulis(10)
                >>> A = X[0]*Y[8] + .25 * Z[9]
        )DOC");

    m.def("fermis",&fermis,R"DOC(
            Returns annihilation operators

            Example:
                >>> from fastfermion import fermis
                >>> f = fermis(10)
                >>> B = f[0].dagger() * f[0] - f[3]
        )DOC");

    m.def("majoranas",&majoranas,R"DOC(
            Returns Majorana operators

            Example:
                >>> from fastfermion import majoranas
                >>> m = majoranas(10)
                >>> C = m[0]*m[9] - m[8]
        )DOC");

    m.def("paulistrings", py::overload_cast<int>(&paulistrings));
    m.def("paulistrings", py::overload_cast<int, int>(&paulistrings));
    m.def("paulistrings", py::overload_cast<int, int, std::function<bool(const std::vector<int>&, const std::vector<char>&)>>(&paulistrings));

    m.def("fermistrings", py::overload_cast<int>(&fermistrings));
    m.def("fermistrings", py::overload_cast<int, int>(&fermistrings));
    m.def("fermistrings", py::overload_cast<int, int, std::function<bool(const std::vector<int>&, const std::vector<int>&)>>(&fermistrings));

    m.def("majoranastrings", py::overload_cast<int>(&majoranastrings));
    m.def("majoranastrings", py::overload_cast<int, int>(&majoranastrings));
    m.def("majoranastrings", py::overload_cast<int, int, std::function<bool(const std::vector<int>&)>>(&majoranastrings));
}

PYBIND11_MODULE(ffcore, m, py::mod_gil_not_used()) {

    // The module name (ffcore) is given as the first macro argument (it should not be in quotes).
    // The second argument (m) defines a variable of type py::module_ which is the main interface for creating bindings.

    m.attr("MAX_QUBITS") = ff_ulong::DIGITS;
    m.attr("FERMI_SYMBOL") = ff_config.fermi_symbol;
    m.attr("MAJORANA_SYMBOL") = ff_config.majorana_symbol;
    m.attr("DAGGER_SYMBOL") = ff_config.dagger_symbol;

    #ifdef FF_VERSION
        m.attr("__version__") = FF_VERSION;
    #else
        m.attr("__version__") = "";
    #endif
    
    py::class_<PauliString> pyPauliString = py::class_<PauliString>(m, "PauliString")
        .def(py::init<>())
        .def(py::init<const std::string&>()) 
        .def(py::init<const std::vector<std::pair<int,char>>&>())
        .def("permute", &PauliString::permute)
        .def("to_string", &PauliString::to_string, py::arg("n") = 0)
        .def("indices", &PauliString::indices)
        .def("degree", [](const PauliString& a, const std::optional<char>& v) { return v.has_value() ? a.degree(v.value()) : a.degree(); }, py::arg("v") = py::none())
        .def("extent", &PauliString::extent)
        .def("commutes", py::overload_cast<const PauliString&>(&PauliString::commutes, py::const_))
        .def("commutes", py::overload_cast<const PauliPolynomial&>(&PauliString::commutes, py::const_))
        .def("commutator", [](const PauliString& a, const PauliString& b) { return PauliPolynomial(a.commutator(b)); })
        .def("commutator", py::overload_cast<const PauliPolynomial&>(&PauliString::commutator, py::const_))
        .def("__str__", &PauliString::to_compact_string)
        .def("__repr__", &PauliString::to_compact_string)
        .def("__hash__", &PauliString::hash)

        // .def("sparse", [](const PauliString& a) { return CSCMatrix_to_scipy(a.sparse()); })
        // .def("sparse", [](const PauliString& a, int n) { return CSCMatrix_to_scipy(a.sparse(n)); })
        
        .def("tofermi",[](const PauliString& p) { return reverse_jordan_wigner(p); })
        .def("tomajorana",[](const PauliString& p) { return MajoranaPolynomial(pauli_to_majorana(p)); })

        .def("__eq__", [](const PauliString &a, const ff_complex& b) { return a == b; })
        .def("__eq__", [](const PauliString &a, const PauliString& b) { return a == b; })

        .def("__add__", [](const PauliString &a, ff_complex b) { return b == ff_complex(0,0) ? a : (a+PauliPolynomial(b)); })
        .def("__add__", [](const PauliString &a, const PauliString& b) { return a+b; })
        .def("__add__", [](const PauliString &a, const PauliPolynomial& b) { return a+b; })

        .def("__sub__", [](const PauliString &a, ff_complex b) { return b == ff_complex(0,0) ? a : (a+PauliPolynomial(-b)); })
        .def("__sub__", [](const PauliString &a, const PauliString& b) { return a-b; })
        .def("__sub__", [](const PauliString &a, const PauliPolynomial& b) { return a-b; })

        .def("__mul__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(a*b); })
        .def("__mul__", [](const PauliString &a, const PauliString& b) { return PauliPolynomial(a*b); })
        .def("__mul__", [](const PauliString &a, const PauliPolynomial& b) { return PauliPolynomial(a*b); })

        .def("__rsub__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(b == ff_complex(0,0) ? -a : (PauliPolynomial(b)-a)); })
        .def("__radd__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(b == ff_complex(0,0) ? a : (PauliPolynomial(b)+a)); })
        .def("__rmul__", [](const PauliString &a, ff_complex b) { return PauliPolynomial(b*a); });


    py::class_<PauliPolynomial> pyPauliPolynomial = py::class_<PauliPolynomial>(m, "PauliPolynomial")
        .def(py::init<>()) 
        .def(py::init<const ff_complex&>())
        .def(py::init<const PauliString&>())
        .def(py::init<const PauliMonomial&>())
        .def(py::init<const std::vector<std::pair<int,char>>&>())
        .def(py::init<const std::vector<std::pair<int,char>>&, const ff_complex&>())
        .def(py::init<const PauliPolynomial&>())
        .def("degree", [](const PauliPolynomial& a, const std::optional<char>& v) { return v.has_value() ? a.degree(v.value()) : a.degree(); }, py::arg("v") = py::none())
        .def("dagger", &PauliPolynomial::dagger)
        .def("coefficient", py::overload_cast<const PauliString&>(&PauliPolynomial::coefficient, py::const_))
        .def("coefficient", [](const PauliPolynomial& a, const std::vector<std::pair<int,char>>& indices) { return a.coefficient(PauliString(indices)); })
        .def("coefficient", [](const PauliPolynomial& a, const std::string& ps) { return a.coefficient(PauliString(ps)); })
        .def("support", &PauliPolynomial::support_set)
        /*.def("sparse", [](const PauliPolynomial& a) { return CSCMatrix_to_scipy(a.sparse()); })
        .def("sparse", [](const PauliPolynomial& a, int n) { return CSCMatrix_to_scipy(a.sparse(n)); })
        .def("sparse", [](const PauliPolynomial& a, int n, int nup) { return CSCMatrix_to_scipy(a.sparse(n, nup)); })*/
        .def("overlapwithzero", &PauliPolynomial::overlapwithzero)
        .def("tofermi",[](const PauliPolynomial& p) { return reverse_jordan_wigner(p); })
        .def("tomajorana",[](const PauliPolynomial& p) { return pauli_to_majorana(p); });

    add_pauli_sparse(pyPauliString);
    add_pauli_sparse(pyPauliPolynomial);

    add_poly_basic_ops<PauliPolynomial, PauliString, PauliMonomial>(pyPauliPolynomial);

    py::class_<FermiString> pyFermiString = py::class_<FermiString>(m, "FermiString")
        .def(py::init<>())
        .def(py::init<const std::vector<std::pair<int,bool>>&>())
        .def(py::init<const std::vector<int>&, const std::vector<int>&>())
        .def("extent", &FermiString::extent)
        .def("indices", &FermiString::indices)
        .def("__hash__", &FermiString::hash)
        .def("degree", [](const FermiString& a, const std::optional<int>& v) {
            if(v.has_value()) {
                return a.degree(v.value());
            } else {
                return a.degree();
            }
        }, py::arg("v") = py::none())
        .def("permute", [](const FermiString& a, const std::vector<int>& perm) { return FermiPolynomial(a.permute(perm)); })
        .def("commutes", py::overload_cast<const FermiString&>(&FermiString::commutes, py::const_))
        .def("commutes", py::overload_cast<const FermiPolynomial&>(&FermiString::commutes, py::const_))
        .def("commutator", py::overload_cast<const FermiString&>(&FermiString::commutator, py::const_))
        .def("commutator", py::overload_cast<const FermiPolynomial&>(&FermiString::commutator, py::const_))
        //.def("cre", [](const FermiString& a) { return a.cre.rsupport(); })
        //.def("ann", [](const FermiString& a) { return a.ann.rsupport(); })
        .def("dagger", [](const FermiString& a) { return FermiPolynomial(a.dagger()); })
        // .def("sparse", [](const FermiString& a) { return CSCMatrix_to_scipy(a.sparse()); })
        // .def("sparse", [](const FermiString& a, int n) { return CSCMatrix_to_scipy(a.sparse(n)); })
        .def("topauli",[](const FermiString& p) { return jordan_wigner(p); })
        .def("tomajorana",[](const FermiString& p) { return fermi_to_majorana(p); })
        .def("__eq__", [](const FermiString &a, const ff_complex& b) { return a == b; })
        .def("__eq__", [](const FermiString &a, const FermiString& b) { return a == b; })

        .def("__neg__", [](const FermiString &a) { return FermiPolynomial(-a); })
        .def("__div__", [](const FermiString &a, const ff_complex& b) { return FermiPolynomial(a*(1.0/b)); })

        .def("__add__", [](const FermiString &a, const ff_complex& b) { return a+b; })
        .def("__add__", [](const FermiString &a, const FermiString& b) { return a+b; })

        .def("__sub__", [](const FermiString &a, const ff_complex& b) { return a-b; })
        .def("__sub__", [](const FermiString &a, const FermiString& b) { return a-b; })

        .def("__mul__", [](const FermiString &a, const ff_complex& b) { return FermiPolynomial(a*b); })
        .def("__mul__", [](const FermiString &a, const FermiString& b) { return FermiPolynomial(a*b); })
        .def("__mul__", [](const FermiString &a, const FermiPolynomial& b) { return FermiPolynomial(a*b); })
        
        .def("__rsub__", [](const FermiString &a, const ff_complex& b) { return b-a; })
        .def("__radd__", [](const FermiString &a, const ff_complex& b) { return b+a; })
        .def("__rmul__", [](const FermiString &a, const ff_complex& b) { return FermiPolynomial(b*a); })

        .def("__str__", &FermiString::to_compact_string)
        .def("__repr__", &FermiString::to_compact_string);


    py::class_<FermiPolynomial> pyFermiPolynomial = py::class_<FermiPolynomial>(m, "FermiPolynomial")
        .def(py::init<>())
        .def(py::init<const ff_complex&>())
        .def(py::init<const FermiString&>())
        .def(py::init<const FermiPolynomial&>())
        .def(py::init<const std::vector<std::pair<int,bool>>&>())
        .def(py::init<const std::vector<std::pair<int,bool>>&, const ff_complex&>())
        .def(py::init<const std::vector<int>&, const std::vector<int>&, ff_complex>())
        .def("degree", [](const FermiPolynomial& a, const std::optional<int>& v) { return v.has_value() ? a.degree(v.value()) : a.degree(); }, py::arg("v") = py::none())
        .def("dagger", &FermiPolynomial::dagger)
        .def("coefficient", py::overload_cast<const FermiString&>(&FermiPolynomial::coefficient, py::const_))
        .def("coefficient", [](const FermiPolynomial& a, const std::string& fs) { return a.coefficient(FermiString(fs)); })
        .def("coefficient", [](const FermiPolynomial& a, const std::vector<std::pair<int,bool>>& fs) { return a.coefficient(FermiString(fs)); })
        .def("support", &FermiPolynomial::support_set)
        .def("overlapwithvacuum", &FermiPolynomial::overlapwithvacuum)
        .def("topauli",[](const FermiPolynomial& p) { return jordan_wigner(p); })
        .def("tomajorana",[](const FermiPolynomial& p) { return fermi_to_majorana(p); });

    
    add_fermi_sparse(pyFermiString);
    add_fermi_sparse(pyFermiPolynomial);

    add_poly_basic_ops<FermiPolynomial, FermiString, FermiMonomial>(pyFermiPolynomial);

    py::class_<MajoranaString> pyMajoranaString = py::class_<MajoranaString>(m, "MajoranaString")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&>())
        .def("extent", &MajoranaString::extent)
        .def("degree", &MajoranaString::degree)
        .def("unpaired", &MajoranaString::unpaired)
        .def("indices", &MajoranaString::support_set)
        .def("is_hermitian", &MajoranaString::is_hermitian)
        .def("commutes", py::overload_cast<const MajoranaString&>(&MajoranaString::commutes, py::const_))
        .def("commutes", py::overload_cast<const MajoranaPolynomial&>(&MajoranaString::commutes, py::const_))
        .def("commutator", [](const MajoranaString& a, const MajoranaString& b) { return MajoranaPolynomial(a.commutator(b)); })
        .def("commutator", py::overload_cast<const MajoranaPolynomial&>(&MajoranaString::commutator, py::const_))
        .def("permute", [](const MajoranaString& a, const std::vector<int>& perm) { return MajoranaPolynomial(a.permute(perm)); })
        .def("dagger", [](const MajoranaString& a) { return MajoranaPolynomial(a.dagger()); })
        // .def("sparse", [](const MajoranaString& a) { return CSCMatrix_to_scipy(a.sparse()); })
        .def("topauli",[](const MajoranaString& p) { return jordan_wigner(p); })
        .def("tofermi",[](const MajoranaString& p) { return majorana_to_fermi(p); })

        .def("__hash__", [](const MajoranaString &a) { return a.hash(); })

        .def("__eq__", [](const MajoranaString &a, const ff_complex& b) { return a == b; })
        .def("__eq__", [](const MajoranaString &a, const MajoranaString& b) { return a == b; })

        .def("__neg__", [](const MajoranaString &a) { return MajoranaPolynomial(-a); })
        .def("__div__", [](const MajoranaString &a, const ff_complex& b) { return MajoranaPolynomial(a*(1.0/b)); })

        .def("__add__", [](const MajoranaString &a, const ff_complex& b) { return a+b; })
        .def("__add__", [](const MajoranaString &a, const MajoranaString& b) { return a+b; })
        .def("__add__", [](const MajoranaString &a, const MajoranaPolynomial& b) { return a+b; })

        .def("__sub__", [](const MajoranaString &a, const ff_complex& b) { return a-b; })
        .def("__sub__", [](const MajoranaString &a, const MajoranaString& b) { return a-b; })
        .def("__sub__", [](const MajoranaString &a, const MajoranaPolynomial& b) { return a-b; })

        .def("__mul__", [](const MajoranaString &a, const ff_complex& b) { return MajoranaPolynomial(a*b); })
        .def("__mul__", [](const MajoranaString &a, const MajoranaString& b) { return MajoranaPolynomial(a*b); })
        .def("__mul__", [](const MajoranaString &a, const MajoranaPolynomial& b) { return MajoranaPolynomial(a*b); })
        
        .def("__rsub__", [](const MajoranaString &a, const ff_complex& b) { return b-a; })
        .def("__radd__", [](const MajoranaString &a, const ff_complex& b) { return b+a; })
        .def("__rmul__", [](const MajoranaString &a, const ff_complex& b) { return MajoranaPolynomial(b*a); })

        .def("__str__", &MajoranaString::to_compact_string)
        .def("__repr__", &MajoranaString::to_compact_string);

    py::class_<MajoranaPolynomial> pyMajoranaPolynomial = py::class_<MajoranaPolynomial>(m, "MajoranaPolynomial")
        .def(py::init<>())
        .def(py::init<const ff_complex&>())
        .def(py::init<const MajoranaString&>())
        .def(py::init<const MajoranaPolynomial&>())
        .def(py::init<const std::vector<int>&>())
        .def(py::init<const std::vector<int>&, ff_complex>())
        .def("degree", &MajoranaPolynomial::degree)
        .def("dagger", &MajoranaPolynomial::dagger)
        .def("coefficient", py::overload_cast<const MajoranaString&>(&MajoranaPolynomial::coefficient, py::const_))
        .def("coefficient", [](const MajoranaPolynomial& a, const std::string& ms) { return a.coefficient(MajoranaString(ms)); })
        .def("coefficient", [](const MajoranaPolynomial& a, const std::vector<int>& ms) { return a.coefficient(MajoranaString(ms)); })
        .def("support", &MajoranaPolynomial::support_set)
        // .def("sparse", [](const MajoranaPolynomial& a) { return CSCMatrix_to_scipy(a.sparse()); })
        .def("topauli",[](const MajoranaPolynomial& a) { return jordan_wigner(a); })
        .def("tofermi",[](const MajoranaPolynomial& a) { return majorana_to_fermi(a); });

    add_poly_basic_ops<MajoranaPolynomial, MajoranaString, MajoranaMonomial>(pyMajoranaPolynomial);

    add_majorana_sparse(pyMajoranaString);
    add_majorana_sparse(pyMajoranaPolynomial);

    // UTILS

    add_gen(m);
    add_transforms(m);
    add_states(m);
    
    add_pauli_gates(m);
    add_propagation_common(m);
    add_pauli_propagation(m);

    add_majorana_propagation(m);
    
}

}