#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

void bind_cosserat_rod(pybind11::module& m);
void bind_parallel_robot(pybind11::module& m);
void bind_tendon_robot(pybind11::module& m);