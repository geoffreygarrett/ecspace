//
// Created by ggarrett on 11/30/22.
//

#ifndef TUDAT_ENTT_SYSTEMS_H
#define TUDAT_ENTT_SYSTEMS_H

#include "accelerations.h"
#include "components.h"
#include "entt/entt.hpp"
#include <execution>


class PrintSystem {
public:
    void update(entt::registry &registry) {
        registry.view<name, position, velocity, acceleration>().each([](auto entity, auto &name, auto &position, auto &velocity, auto &acceleration) {
            std::cout << name << " is at (" << position.x << ", " << position.y << ", " << position.z << ") with velocity (" << velocity.dx << ", " << velocity.dy << ", " << velocity.dz << ") and acceleration (" << acceleration.ddx << ", " << acceleration.ddy << ", " << acceleration.ddz << ")" << std::endl;
        });
    }
};


// gets all bodies states who are being propagated and form a contiguous block
// in the state vector
class StateSystem {
public:
    // return a vector of pointers to the state values of the bodies
    static std::vector<double *> state_pointer_vector(entt::registry &registry) {

        // get all entities that are being propagated
        auto view = registry.view<position, velocity, acceleration>();

        // create vector of pointers to the state values
        std::vector<double *> state_pointer_vector;

        // loop over all entities
        for (auto entity: view) {

            // get the position of the entity
            auto &position_ = registry.get<position>(entity);

            // get the velocity of the entity
            auto &velocity_ = registry.get<velocity>(entity);

            // get the acceleration of the entity
            auto &acceleration_ = registry.get<acceleration>(entity);
            state_pointer_vector.push_back(&position_.x);
            state_pointer_vector.push_back(&position_.y);
            state_pointer_vector.push_back(&position_.z);
            state_pointer_vector.push_back(&velocity_.dx);
            state_pointer_vector.push_back(&velocity_.dy);
            state_pointer_vector.push_back(&velocity_.dz);
            state_pointer_vector.push_back(&acceleration_.ddx);
            state_pointer_vector.push_back(&acceleration_.ddy);
            state_pointer_vector.push_back(&acceleration_.ddz);
        }

        return state_pointer_vector;
    }
};

class PhysicsSystem {
public:
    explicit PhysicsSystem(): parallel_(true) {}

    void update(entt::registry &registry, double dt) const {
        if (parallel_){
//            auto view = registry.view<position, velocity, const acceleration>();
            auto posvelview = registry.view<position, const velocity>();
            auto velaccview = registry.view<velocity, const acceleration>();
            std::for_each(std::execution::par_unseq, posvelview.begin(), posvelview.end(), [&registry, &dt](auto entity) {
                auto &position_ = registry.get<position>(entity);
                auto &velocity_ = registry.get<velocity>(entity);
                position_.x += velocity_.dx * dt;
                position_.y += velocity_.dy * dt;
                position_.z += velocity_.dz * dt;
            });
            std::for_each(std::execution::par_unseq, velaccview.begin(), velaccview.end(), [&registry, &dt](auto entity) {
                auto &velocity_ = registry.get<velocity>(entity);
                auto &acceleration_ = registry.get<acceleration>(entity);
                velocity_.dx += acceleration_.ddx * dt;
                velocity_.dy += acceleration_.ddy * dt;
                velocity_.dz += acceleration_.ddz * dt;
            });

//            std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&registry, &dt](auto entity) {
//                auto &position_ = registry.get<position>(entity);
//                auto &velocity_ = registry.get<velocity>(entity);
//                auto &acceleration_ = registry.get<acceleration>(entity);
//                position_.x += velocity_.dx * dt;
//                position_.y += velocity_.dy * dt;
//                position_.z += velocity_.dz * dt;
//                velocity_.dx += acceleration_.ddx * dt;
//                velocity_.dy += acceleration_.ddy * dt;
//                velocity_.dz += acceleration_.ddz * dt;
//            });

        } else {
            auto view = registry.view<position, velocity, acceleration>();
            for (auto entity: view) {
                auto &position_ = registry.get<position>(entity);
                auto &velocity_ = registry.get<velocity>(entity);
                auto &acceleration_ = registry.get<acceleration>(entity);
                position_.x += velocity_.dx * dt;
                position_.y += velocity_.dy * dt;
                position_.z += velocity_.dz * dt;
                velocity_.dx += acceleration_.ddx * dt;
                velocity_.dy += acceleration_.ddy * dt;
                velocity_.dz += acceleration_.ddz * dt;
            }
        }




    }

private:
    double dt_;
    const bool parallel_;
};


#include "../src/ecspace/core/twobody.h"
#include "consts.h"
#include <cmath>
#include <filesystem>

#if ECSPACE_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

class MatplotlibSystem {
public:
    explicit MatplotlibSystem(entt::registry &registry) {
        //        plt::clf();
        //        plt::xlabel("x [AU]");
        //        plt::ylabel("y [AU]");
        //        plt::title("Solar System");
        //        plt::axis("equal");
        ////        plt::xlim(-1.5, 1.5);
        ////        plt::ylim(-1.5, 1.5);
        //        plt::grid(true);
        //        registry.view<name, position, simulated>().each([&](auto entity, auto name, auto &position, auto &simulated) {
        //            // store initial position
        //            struct position transformed_position = transform(registry, position);
        //            x[entity].push_back(transformed_position.x);
        //            y[entity].push_back(transformed_position.y);
        //        });
    }

    void initialize(entt::registry &registry) {
        registry.view<name, position, simulated>().each([&](auto entity, auto name, auto &position, auto &simulated) {
            // store initial position
            struct position transformed_position = transform(registry, position);
            x[entity].push_back(transformed_position.x);
            y[entity].push_back(transformed_position.y);
        });


        // if reference_frame has gravitational parameter and a parent with a gravitational parameter, then
        // plot the sphere of influence
        auto mu_ptr = registry.try_get<gravitational_parameter>(reference_frame);
        if (mu_ptr != nullptr) {
            auto mu = *mu_ptr;
            auto parent_ptr = registry.try_get<parent>(reference_frame);
            if (parent_ptr != nullptr) {
                auto parent = *parent_ptr;
                auto parent_mu_ptr = registry.try_get<gravitational_parameter>(parent);
                auto parent_position_ptr = registry.try_get<position>(parent);
                if (parent_mu_ptr != nullptr && parent_position_ptr != nullptr) {
                    auto parent_mu = *parent_mu_ptr;
                    auto parent_position = *parent_position_ptr;
                    auto pos = registry.get<position>(reference_frame);
                    auto distance = std::sqrt(
                            std::pow(pos.x - parent_position.x, 2) +
                            std::pow(pos.y - parent_position.y, 2) +
                            std::pow(pos.z - parent_position.z, 2));
                    auto radius = r_soi(distance, mu, parent_mu);
                    for (double theta = 0; theta < 2 * M_PI; theta += M_PI / 100) {
                        soi_x.push_back(radius * std::cos(theta) / scale);
                        soi_y.push_back(radius * std::sin(theta) / scale);
                    }
                    plt::plot(soi_x, soi_y, "k--");
                }
            }
        }
    }

    void set_reference_frame(entt::entity entity) {
        reference_frame = entity;
    }

    void set_scale(double scale) {
        scale = scale;
    }

    position transform(entt::registry &registry, position pos) {
        if (reference_frame == entt::null) {
            return position{pos.x / scale, pos.y / scale, pos.z / scale};
        } else {
            auto &reference_position = registry.get<position>(reference_frame);
            return position{
                    (pos.x - reference_position.x) / scale,
                    (pos.y - reference_position.y) / scale,
                    (pos.z - reference_position.z) / scale};
        }
    }

    void update(entt::registry &registry) {

        int num = 50;
        registry.view<name, position, simulated>().each([&](auto entity, auto &name, auto &position, auto &simulated) {
            // store initial position
            struct position transformed_position = transform(registry, position);
            x[entity].push_back(transformed_position.x);
            y[entity].push_back(transformed_position.y);
        });

        if (i % 5 == 0) {
            plt::clf();
            plt::xlabel("x [AU]");
            plt::ylabel("y [AU]");
            plt::title("Solar System");
            plt::axis("equal");
            plt::grid(true);
            for (auto entity: registry.view<name, position, simulated>()) {
                auto &n = registry.get<name>(entity);
                // plot last 1000 points
                auto &x_vec = x[entity];
                auto &y_vec = y[entity];
                if (x_vec.size() > num) {
                    x_vec.erase(x_vec.begin(), x_vec.begin() + x_vec.size() - num);
                    y_vec.erase(y_vec.begin(), y_vec.begin() + y_vec.size() - num);
                }
                plt::named_plot(n, x_vec, y_vec);
//                plt::named_plot(n, x[entity], y[entity]);
            }
//            plot soi
            plt::plot(soi_x, soi_y, "k--");
            plt::legend();
            plt::pause(0.000001);
        }
        i++;

        // save image
//        if (i % 10 == 0) {
//            plt::clf();
//            plt::xlabel("x [AU]");
//            plt::ylabel("y [AU]");
//            plt::title("Solar System");
//            plt::axis("equal");
//            plt::grid(true);
//            for (auto entity: registry.view<name, position, simulated>()) {
//                auto &n = registry.get<name>(entity);
//                // plot last 1000 points
//                auto &x_vec = x[entity];
//                auto &y_vec = y[entity];
//                if (x_vec.size() > num) {
//                    x_vec.erase(x_vec.begin(), x_vec.begin() + x_vec.size() - num);
//                    y_vec.erase(y_vec.begin(), y_vec.begin() + y_vec.size() - num);
//                }
//                plt::named_plot(n, x_vec, y_vec);
//                //                plt::named_plot(n, x[entity], y[entity]);
//            }
//            //            plot soi
//            // make images directory
//            std::filesystem::create_directory("images");
//            plt::plot(soi_x, soi_y, "k--");
//            // pad filename with zeros up to 5 digits
//            std::stringstream ss;
//            ss << std::setw(5) << std::setfill('0') << i;
//            std::string filename = "images/" + ss.str() + ".png";
//            plt::save(filename);
//        }

    }

private:
    std::map<entt::entity, std::vector<double>> x{};
    std::map<entt::entity, std::vector<double>> y{};

    // soi x and y
    std::vector<double> soi_x{};
    std::vector<double> soi_y{};

    entt::entity reference_frame{entt::null};
    double scale = 1;
    int i = 0;
};
#endif // ECSPACE_MATPLOTLIB


// TODO: Figure out a way to handle uninitialized values.
//       For example, I faced a problem here that 'acceleraion' components
//       had not been assigned to entities, so retrieving them caused a crash
//       for the initial state. I solved this by initializing the acceleration
class CSVWriterSystem {
    /// writes <epoch> <name_of_entity1>_<position>_x, <name_of_entity1>_<position>_y, <name_of_entity1>_<position>_z,
    ///        <name_of_entity1>_<velocity>_x, <name_of_entity1>_<velocity>_y, <name_of_entity1>_<velocity>_z,
    ///        <name_of_entity1>_<acceleration>_x, <name_of_entity1>_<acceleration>_y, <name_of_entity1>_<acceleration>_z
    ///        <name_of_entity2>_<position>_x, <name_of_entity2>_<position>_y, <name_of_entity2>_<position>_z ...
    /// to a csv file
public:
    explicit CSVWriterSystem(entt::registry &registry) {
        // get all entities with a name, position, velocity and acceleration
        auto view = registry.view<name, position, velocity, acceleration, simulated>();

        // open the file
        std::ofstream file;
        file.open("output.csv");

        // iterate over all entities with a name, position, velocity and acceleration
        for (auto entity: view) {
            // get the name, position, velocity and acceleration of the entity
            auto &name_ = registry.get<name>(entity);
            auto &position_ = registry.get<position>(entity);
            auto &velocity_ = registry.get<velocity>(entity);
            auto &acceleration_ = registry.get<acceleration>(entity);

            // add the name, position, velocity and acceleration to the csv file
            file << name_ << "_position_x," << name_ << "_position_y," << name_ << "_position_z,"
                 << name_ << "_velocity_x," << name_ << "_velocity_y," << name_ << "_velocity_z,"
                 << name_ << "_acceleration_x," << name_ << "_acceleration_y," << name_ << "_acceleration_z,";
        }


        // add a new line to the csv file
        file << std::endl;

        // close the file
        file.close();
    }

    void update(entt::registry &registry) {
        // get all entities with a name, position, velocity and acceleration
        // get all entities with a name, position, velocity and acceleration
        auto view = registry.view<name, position, velocity, acceleration, simulated>();

        // open file
        std::ofstream file;
        file.open("output.csv", std::ios_base::app);


        // iterate over all entities with a name, position, velocity and acceleration
        for (auto entity: view) {
            // get the name, position, velocity and acceleration of the entity
            auto &name_ = registry.get<name>(entity);
            auto &position_ = registry.get<position>(entity);
            auto &velocity_ = registry.get<velocity>(entity);
            auto &acceleration_ = registry.get<acceleration>(entity);

            // add the name, position, velocity and acceleration to the csv file
            file << position_.x << "," << position_.y << "," << position_.z << ","
                 << velocity_.dx << "," << velocity_.dy << "," << velocity_.dz << ","
                 << acceleration_.ddx << "," << acceleration_.ddy << "," << acceleration_.ddz << ",";
        }

        // add a new line to the csv file
        file << std::endl;

        // close the file
        file.close();
    }
};


#endif//TUDAT_ENTT_SYSTEMS_H
