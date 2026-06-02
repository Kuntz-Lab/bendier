# BENDIER: Bayesian Estimation of Nonlinear Deformation: Inference for Elastic Robots

## Description

Continuum robot state estimation can be formulated similarly to SLAM, where variables are connected through spatial motion priors and measurement factors.
This repository demonstrates how to construct factor graph representations of the conditional distribution over continuum robot configurations.

We leverage the sparse nonlinear optimization capabilities of GTSAM to efficiently estimate continuum robot states.
The repository supports:
- `bendier_solvers`: A standalone static C++ library implementing factor graph optimization methods
- `bendier`: Python package that bundles solver bindings and plotters under one import

If you use this code, please cite our RAL paper (see preprint version [here](https://arxiv.org/abs/2601.04493)):

```
@article{ferguson2026continuum,
  title   = {Continuum Robot State Estimation with Actuation Uncertainty},
  author  = {Ferguson, James M. and Kuntz, Alan and Hermans, Tucker},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {xx},
  number  = {x},
  pages   = {xxx--xxx},
  year    = {2026},
  doi     = {xxxxxxxxxxx}
}
```

TODO: Add citation metadata once the paper is published.

## Build/Install Dependencies

### Install Eigen3

Eigen3 is required for numerical linear algebra operations in `bendier` and `gtsam`.
You likely already have it installed, but if not, you can install with:

```bash
sudo apt update
sudo apt install -y libeigen3-dev
```

We have tested with Eigen3 version: **3.4.0**.

### Build/Install GTSAM

`bendier` dynamically loads classes from GTSAM, so GTSAM must first be built and installed.
The best way to do this is from source, which allows you to ensure that all required dependencies are properly configured.

First clone GTSAM and create a build directory:

```bash
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.3a1 # Tested GTSAM version
mkdir build 
cd build
```

Next configure the build with CMake:

```bash
cmake .. -DGTSAM_USE_SYSTEM_EIGEN=ON
```

Use system Eigen for GTSAM (`-DGTSAM_USE_SYSTEM_EIGEN=ON`) so both GTSAM and `bendier` resolve Eigen from the same system installation. This avoids accidentally mixing GTSAM's internal vendored Eigen headers with the system Eigen headers.

At this point, verify that CMake found all required dependencies (e.g., Boost, Eigen, TBB).
Ensure that there are no critical warnings during configuration.
For more information on dependencies, see the GTSAM [installation documentation](https://borglab.github.io/gtsam/install/)

If everything looks good, you can now build and install gtsam, which will take several minutes:

```bash
make -j8
sudo make install
```

This installs GTSAM headers (needed to *build* `bendier`) in `/usr/local/include` and library files (needed to *run* `bendier`) in `/usr/local/lib`.

## Build/Install `bendier` python package

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

Note that we have tested with **Python 3.12.3** but expect that other versions should work as well.

Install the python dependencies required for plotting and simulation:
```bash
pip install -r requirements.txt
```

Now build and install the `bendier` Python package:
```bash
pip install . -v
```

The final output should include `Successfully installed bendier`, meaning that you can now `import bendier` in Python when the virtual environment is active.

## Build C++ library Only

```bash
cmake -S . -B build-cpp -DBENDIER_BUILD_PYTHON=OFF
cmake --build build-cpp -j
cmake --install build-cpp --prefix build-cpp/install
```

This produces a standalone `bendier_solvers` C++ library.
The last line creates a staged install under `build-cpp/install/` that you can copy or point other projects at.

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

When the chosen script runs successfully, a PyVista render window will appear showing real-time solution geometries for the selected model.
Solution metadata is displayed in the upper-right corner, including optimization solve times and related diagnostics.

## Debugging Tips

1. You may get something like: `ImportError: libgtsam.so.4: cannot open shared object file: No such file or directory`.
This indicates that the GTSAM library installation directory is not visible to the dynamic linker.
In most cases this can be resolved by running `sudo ldconfig` and then rerunning the script.

2. If plotting is slow, first check which GPU is actually rendering OpenGL. On Ubuntu this is usually the most useful single command:

```bash
glxinfo -B
```

Look for `OpenGL vendor string` and `OpenGL renderer string`:
- `llvmpipe`, `softpipe`, or `OSMesa` means software rendering.
- `Intel` means the integrated GPU is active.
- `NVIDIA` means the discrete NVIDIA GPU is active.

To switch the machine to NVIDIA system-wide (careful!), use PRIME Select and reboot:

```bash
sudo prime-select nvidia
sudo reboot
```

If you do not want to change the whole session, you can run like this to select the GPU on a per-command basis:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python -m tests.parallel_robot.test_simple
```