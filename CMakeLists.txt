cmake_minimum_required(VERSION 3.18)
project(bendier LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for clangd" FORCE)

option(BENDIER_BUILD_PYTHON "Build Python bindings" ON)

# Set release build flags for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

set(BENDIER_SOLVER_SOURCES
  src/cosserat_rod/BoundaryStressFactor.cpp
  src/cosserat_rod/CosseratRodModel.cpp
  src/cosserat_rod/CosseratRodSolver.cpp
  src/cosserat_rod/CosseratStrainFactor.cpp
  src/cosserat_rod/CosseratStressFactor.cpp
  src/measurement/ActuationForceMeasFactor.cpp
  src/parallel_robot/ParallelRobotModel.cpp
  src/parallel_robot/ParallelRobotSolver.cpp
  src/parallel_robot/PlatformWrenchBalanceFactor.cpp
  src/parallel_robot/SingleRodBaseFactor.cpp
  src/tendon_robot/TendonDiscWrenchFactor.cpp
  src/tendon_robot/TendonRobotModel.cpp
  src/tendon_robot/TendonRobotSolver.cpp
  src/utils/WrenchTransforms.cpp
)

# Static solver library keeps pip builds predictable. We can add a shared option later.
add_library(bendier_solvers STATIC ${BENDIER_SOLVER_SOURCES})

target_include_directories(bendier_solvers
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GTSAM_INCLUDE_DIR}
)

set_target_properties(bendier_solvers PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(bendier_solvers PUBLIC gtsam Eigen3::Eigen)

install(TARGETS bendier_solvers
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cosserat_rod
  ${CMAKE_CURRENT_SOURCE_DIR}/src/measurement
  ${CMAKE_CURRENT_SOURCE_DIR}/src/parallel_robot
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tendon_robot
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  DESTINATION include/bendier
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

option(BENDIER_BUILD_TESTS "Build C++ unit tests" ON)

if(BENDIER_BUILD_TESTS)
  find_library(CPPUNITLITE_LIBRARY NAMES CppUnitLite)
  find_path(CPPUNITLITE_INCLUDE_DIR NAMES CppUnitLite/TestHarness.h)

  if(CPPUNITLITE_LIBRARY AND CPPUNITLITE_INCLUDE_DIR)
    enable_testing()

    add_executable(bendier_tests tests/JacobianTests.cpp)

    target_include_directories(bendier_tests
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CPPUNITLITE_INCLUDE_DIR}
        ${GTSAM_INCLUDE_DIR}
    )

    target_link_libraries(bendier_tests PRIVATE bendier_solvers ${CPPUNITLITE_LIBRARY})

    add_test(NAME bendier_tests COMMAND bendier_tests)
  else()
    message(WARNING "CppUnitLite not found; skipping bendier_tests target")
  endif()
endif()

if(BENDIER_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  set(BENDIER_PYBIND_SOURCES
    src/cosserat_rod/pybind.cpp
    src/parallel_robot/pybind.cpp
    src/tendon_robot/pybind.cpp
    src/pybind/bindings.cpp
  )

  pybind11_add_module(_bendier ${BENDIER_PYBIND_SOURCES})

  target_include_directories(_bendier
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${GTSAM_INCLUDE_DIR}
  )

  target_link_libraries(_bendier PRIVATE bendier_solvers)

  install(TARGETS _bendier LIBRARY DESTINATION bendier)
endif()