// File: pybind_hanabi.cpp

#include <pybind11/pybind11.h>
#include "hanabi_lib/canonical_encoders.h" // Include the relevant header

namespace py = pybind11;

PYBIND11_MODULE(pyhanabi, m) {
    py::class_<CanonicalObservationEncoder>(m, "CanonicalObservationEncoder")
        .def(py::init<const HanabiGame&>()) // Constructor
        .def("encode", &CanonicalObservationEncoder::encode)  // Member function
        .def("shape", &CanonicalObservationEncoder::shape); // Expose other methods if necessary
}
