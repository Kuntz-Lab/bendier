#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>


struct Vector4Gaussian {
    gtsam::Vector4 mean;
    gtsam::Matrix4 cov;
};


struct Vector3Gaussian {
    gtsam::Vector3 mean;
    gtsam::Matrix3 cov;
};


struct Vector6Gaussian {
    gtsam::Vector6 mean;
    gtsam::Matrix6 cov;
};


struct Pose3Gaussian {
    gtsam::Matrix4 mean;
    gtsam::Matrix6 cov;
};

