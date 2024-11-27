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
namespace hle = hanabi_learning_env;

PYBIND11_MODULE(hanabi, m) {
    // Binding for HanabiGame class with required argument
    py::class_<hle::HanabiGame>(m, "HanabiGame")
        .def(py::init<const std::unordered_map<std::string, std::string>&>())  // Pass map as argument
        ;
    // Binding for CanonicalObservationEncoder
    py::class_<hle::CanonicalObservationEncoder>(m, "CanonicalObservationEncoder")
        .def(py::init<const hle::HanabiGame*>())  // Provide HanabiGame* as constructor argument
        .def("Shape", &hle::CanonicalObservationEncoder::Shape)
        ;
}