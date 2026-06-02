#pragma once

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Symbol.h>


inline gtsam::SharedDiagonal get_noise_model_rot_pos(double sigma_rot, double sigma_pos) {  //This also shouldnt go here
    gtsam::SharedDiagonal model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
        sigma_rot, sigma_rot, sigma_rot, 
        sigma_pos, sigma_pos, sigma_pos).finished());

    return model;
}


inline void print_values(gtsam::Values values) {
    for (const auto& key_value : values) {
        gtsam::Key key = key_value.key;
        const auto& value = key_value.value;

        std::cout << "Key: " << gtsam::Symbol(key) << std::endl;

        // Use the polymorphic print function of the value
        value.print("Value: ");
    }
}