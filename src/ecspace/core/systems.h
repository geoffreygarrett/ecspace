//
// Created by ggarrett on 11/30/22.
//

#ifndef ECSPACE_SYSTEMS_H
#define ECSPACE_SYSTEMS_H

#include "accelerations.h"
#include "components.h"
#include "entt/entt.hpp"
#include "accelerations.cuh"
#include <execution>

#include "consts.h"
#include "twobody.h"
#include <cmath>
#include <filesystem>


// TODO: Maybe add template based on size of registry.view<const dynamic_influence>();? Is this possible?
class DynamicsSystem {
public:
    typedef int index;

    explicit DynamicsSystem(entt::registry &registry) {
        // assert that all entities with dynamic_influence point to an entity with position, velocity
        // TODO: Should this become an exception/  warning or be handled elsewhere with
        //   a loop which ensures that all entities with dynamic_influence point to an entity with position, velocity?
        //   Perhaps log a warning and attempt to retrieve from... i.e. spice.
        auto view = registry.view<const dynamic_influence>();
        auto count_ = 0;
        for (auto entity: view) {
            auto &dynamic_influence_ = registry.get<dynamic_influence>(entity);
//            std::cout << "Checking dynamic_influence " << registry.get<name>(entity) << std::endl;
            assert(registry.try_get<position>(dynamic_influence_.on_entity) != nullptr);
            assert(registry.try_get<velocity>(dynamic_influence_.on_entity) != nullptr);
            // TODO: Later we will need to consider rotational dynamics.
            // assign state_index and simulated
            auto index_ptr = registry.try_get<index>(dynamic_influence_.on_entity);
            if (index_ptr == nullptr) {
                registry.emplace<index>(dynamic_influence_.on_entity, count_);
                registry.emplace<simulated>(dynamic_influence_.on_entity);
                registry.emplace<acceleration>(dynamic_influence_.on_entity, 0, 0, 0);
                count_++;
            }

            // TODO: Could form dependency graph for entities later... perhaps working towards co-simulation/ or better multi-threading!
            // add influences
            if (registry.try_get<influences>(dynamic_influence_.on_entity) == nullptr) {
                registry.emplace_or_replace<influences>(dynamic_influence_.on_entity);
            } else {
                registry.get<influences>(dynamic_influence_.on_entity).entities.push_back(entity);
            }
//            count_++;
//            std::cout << count_ << std::endl;
        }
        // size state vector
        state_ = Eigen::VectorXd::Zero(6 * count_);

        // size state derivative
        state_derivative_ = Eigen::VectorXd::Zero(6 * count_);

        update(registry);
    }


    void update(entt::registry &registry) {
        // get all entities with a name, position, velocity and acceleration
        auto view = registry.view<index, simulated>();

        // iterate in order of index and update private state_
        for (auto entity: view) {
            auto &index_ = registry.get<index>(entity);
            auto &position_ = registry.get<position>(entity);
            auto &velocity_ = registry.get<velocity>(entity);
            state_(6 * index_) = position_.x;
            state_(6 * index_ + 1) = position_.y;
            state_(6 * index_ + 2) = position_.z;
            state_(6 * index_ + 3) = velocity_.dx;
            state_(6 * index_ + 4) = velocity_.dy;
            state_(6 * index_ + 5) = velocity_.dz;
        }
    }

    Eigen::VectorXd &get_translational_state(entt::registry &registry) {
        return state_;
    }

    void set_translational_state(entt::registry &registry, Eigen::VectorXd state) {
        // get the name, position, velocity and acceleration of the whole system
        auto view = registry.view<position, velocity, simulated>();

        // iterate over all entities with a name, position, velocity and acceleration
        for (auto entity: view) {
            auto &r = registry.get<position>(entity);
            auto &v = registry.get<velocity>(entity);
            auto index_ = registry.get<index>(entity);

            // set the state vector
            r.x = state(6 * index_);
            r.y = state(6 * index_ + 1);
            r.z = state(6 * index_ + 2);
            v.dx = state(6 * index_ + 3);
            v.dy = state(6 * index_ + 4);
            v.dz = state(6 * index_ + 5);
        }
        state_ = state;
    }


    Eigen::VectorXd get_translational_state_derivative(entt::registry &registry, Eigen::VectorXd state, double t) {
        // get all entities with a name, position, velocity and acceleration
        //        auto view = registry.view<name, position, velocity, acceleration, simulated>();
        auto view = registry.view<const simulated>();

        // store current state
        Eigen::VectorXd translational_state_snapshot = get_translational_state(registry);

        // set provided state
        set_translational_state(registry, std::move(state));

        Eigen::VectorXd state_derivative = Eigen::VectorXd::Zero(6 * count_);

        // start timer
//        auto start1 = std::chrono::high_resolution_clock::now();

//#if ECSPACE_CUDA
//
//        calcuate_derivatives<<<1, count_>>>(registry, state_derivative.data(), t);
//
//#else
        // iterate over all simulated bodies
        std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&registry, t](auto entity) {
            // iterate over all dynamic_influences on this body and sum the accelerations
            auto &influences_ = registry.get<influences>(entity);
            //            auto index_ = registry.get<index>(entity);
            auto &acceleration_ = registry.get<acceleration>(entity);
            acceleration total_acceleration{};
            for (auto influence: influences_.entities) {
                auto &dynamic_influence_ = registry.get<dynamic_influence>(influence);
                auto a_ = dynamic_influence_.function(registry, t);
                dynamic_influence_.last_value = a_;
                total_acceleration.ddx += a_[0];
                total_acceleration.ddy += a_[1];
                total_acceleration.ddz += a_[2];
            }
            acceleration_.ddx = total_acceleration.ddx;
            acceleration_.ddy = total_acceleration.ddy;
            acceleration_.ddz = total_acceleration.ddz;


        });
//#endif
        // time
//        auto end1 = std::chrono::high_resolution_clock::now();
//        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
//        std::cout << "Time taken by calculation: " << duration1.count() << " microseconds" << std::endl;

        // start time
//        auto start2 = std::chrono::high_resolution_clock::now();


        // iterate in order of index and update private state_derivative_
        std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&registry, this, t](auto entity) {
            auto &n = registry.get<name>(entity);
            auto &index_ = registry.get<index>(entity);
            auto velocity_ = registry.get<velocity>(entity);
            auto acceleration_ = registry.get<acceleration>(entity);
            // set acceleration component for all entities

            state_derivative_(6 * index_ + 0) = velocity_.dx;
            state_derivative_(6 * index_ + 1) = velocity_.dy;
            state_derivative_(6 * index_ + 2) = velocity_.dz;
            state_derivative_(6 * index_ + 3) = acceleration_.ddx;
            state_derivative_(6 * index_ + 4) = acceleration_.ddy;
            state_derivative_(6 * index_ + 5) = acceleration_.ddz;
        });



        // restore state
        set_translational_state(registry, translational_state_snapshot);

//        auto end2 = std::chrono::high_resolution_clock::now();
//        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
//        std::cout << "Time taken by update: " << duration2.count() << " microseconds" << std::endl;


        return state_derivative_;
    }

private:
    int count_ = 0;
    Eigen::VectorXd state_;
    Eigen::VectorXd state_derivative_;
};


class PrintSystem {
public:
    void update(entt::registry &registry) {
        registry.view<name, position, velocity, acceleration>().each(
                [](auto entity, auto &name, auto &position, auto &velocity, auto &acceleration) {
                    std::cout << name
                              << " is at (" << position.x << ", " << position.y << ", " << position.z
                              << ") with velocity (" << velocity.dx << ", " << velocity.dy << ", " << velocity.dz
                              << ") and acceleration (" << acceleration.ddx << ", " << acceleration.ddy << ", "
                              << acceleration.ddz
                              << ")" << std::endl;
                });
    }
};


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
        if (reference_frame != entt::null) {
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

        int num = 300;
        registry.view<const name, const position, const simulated>().each(
                [&](auto entity, auto &name, auto &position, auto &simulated) {
                    // store initial position
                    struct position transformed_position = transform(registry, position);
                    x[entity].push_back(transformed_position.x);
                    y[entity].push_back(transformed_position.y);
                });

        if (i % 10 == 0) {
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
//            plt::xkcd();

            if (soi_x.size() > 0) {
                plt::plot(soi_x, soi_y, "k--");
            }
//            plt::plot(soi_x, soi_y, "k--");
//            plt::legend();
            plt::pause(0.000001);
        }
        i++;

//        // save image
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
//            ss << std::setw(5) << std::setfill('0') << j;
//            std::string filename = "images/" + ss.str() + ".png";
//            plt::save(filename);
//            j++;
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
    int j = 0;
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


#endif//ECSPACE_SYSTEMS_H
