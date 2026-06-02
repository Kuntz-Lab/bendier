# `bendier`: Bayesian Estimation of Nonlinear Deformation: Inference for Elastic Robots

## Description

Continuum robot state estimation can be formulated similarly to SLAM, where variables are connected through spatial motion priors and measurement factors.
This repository demonstrates how to construct factor graph representations of the conditional distribution over continuum robot configurations.

We leverage the sparse nonlinear optimization capabilities of GTSAM to efficiently estimate continuum robot states.
The repository supports:
- `bendier_solvers`: A standalone static C++ library implementing factor graph optimization methods
- `bendier`: Python package that bundles solver bindings and plotters under one import

## Build/Install GTSAM

The Python bindings dynamically load classes from GTSAM, so GTSAM must first be built and installed from source.

First clone GTSAM and configure the build with CMake:

```bash
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.3a1 # Tested GTSAM version
mkdir build 
cd build
cmake ..
```

At this point, verify that CMake found all required dependencies (e.g., Boost, Eigen, TBB).
Ensure that there are no critical warnings during configuration.
For more information on dependencies, see the GTSAM [installation documentation](https://borglab.github.io/gtsam/install/)

If everything looks good, you can now build and install gtsam, which will take several minutes:

```bash
make -j8
sudo make install
```

This installs GTSAM headers (needed to *build* `bendier`) in `/usr/local/include` and library files (needed to *run* `bendier`) in `/usr/local/lib`.

## Build/Install `bendier`

First clone this repository:
```bash
git clone https://github.com/Kuntz-Lab/bendier.git
cd bendier
```

Next create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the python dependencies required for plotting and simulation:
```bash
pip install -r requirements.txt
```

Now build and install the `bendier` Python package:
```bash
pip install . -v
```

The final output should include `Successfully installed bendier`, meaning that you can now `import bendier` in Python when the virtual environment is active.

## Build C++ only (without pybind)

```bash
cmake -S . -B build-cpp -DBENDIER_BUILD_PYTHON=OFF
cmake --build build-cpp -j
```

This produces a standalone `bendier_solvers` C++ library.

## Run Test Scripts

Plotting utilities live in the `bendier.plotting` package, and runnable examples/tests live in `python/tests/`.
There are several examples you can run to verify the solvers are working.
Here are a few options:

```bash
cd python

python -m tests.cosserat.test_priors_sim
python -m tests.cosserat.spring_sim
python -m tests.tendon_robot.test_simple
python -m tests.parallel_robot.test_simple
```

All RAL paper simulations can be run with 

```bash
bash run_sims.bash
```

You may get an initial error that says something like: `ImportError: libgtsam.so.4: cannot open shared object file: No such file or directory`.
This indicates that the GTSAM library installation directory is not visible to the dynamic linker.
In most cases this can be resolved by running `sudo ldconfig` and then rerunning the script.

When the chosen script runs successfully, a PyVista render window will appear showing real-time solution geometries for the selected model.
Solution metadata is displayed in the upper-right corner, including optimization solve times and related diagnostics.