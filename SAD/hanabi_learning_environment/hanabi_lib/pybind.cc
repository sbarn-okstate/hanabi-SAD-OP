#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To handle STL containers like vectors and maps
#include "hanabi_card.h"    // Include your headers
#include "hanabi_game.h"
#include "hanabi_hand.h"
#include "hanabi_history_item.h"
#include "hanabi_move.h"
#include "hanabi_observation.h"
#include "hanabi_state.h"
#include "util.h"
#include "canonical_encoders.h"

namespace py = pybind11;

PYBIND11_MODULE(hanabi, m) {
    py::class_<CanonicalObservationEncoder>(m, "CanonicalObservationEncoder")
        .def(py::init<>())
        .def("Shape", &CanonicalObservationEncoder::Shape);
}