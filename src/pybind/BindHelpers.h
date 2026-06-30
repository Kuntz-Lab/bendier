#pragma once
#include <pybind11/pybind11.h>
#include "utils/SolverBase.h"

#define BIND_FIELD(Class, field) def_readwrite(#field, &Class::field)

template <typename TMarg>
void bind_solution(pybind11::module& m, const char* name) {
    pybind11::class_<Solution<TMarg>>(m, name)
        .def(pybind11::init<>())
        .BIND_FIELD(Solution<TMarg>, meta)
        .BIND_FIELD(Solution<TMarg>, marginals);
}

template <typename TGaussian, typename TMean, typename TCov>
void bind_gaussian(pybind11::module& m, const char* name) {
    pybind11::class_<TGaussian>(m, name)
        .def(pybind11::init<>())
        .def(pybind11::init<const TMean&, const TCov&>(),
             pybind11::arg("mean"), pybind11::arg("cov"))
        .BIND_FIELD(TGaussian, mean)
        .BIND_FIELD(TGaussian, cov);
}
