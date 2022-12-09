#include "entt/entt.hpp"
#include <chrono>
#include <execution>
#include <fstream>
#include <iostream>

#include "../src/ecspace/core/twobody.h"

#include "../src/ecspace/core/accelerations.h"
#include "../src/ecspace/core/components.h"
#include "../src/ecspace/core/consts.h"
#include "../src/ecspace/core/systems.h"
#include "../src/ecspace/core/termination.h"
#include "../src/ecspace/core/types.h"

#include <Eigen/Dense>
#include <utility>

#include "helper.h"


/// \important
/// - TODO: Employ a sanity check for all accelerations in the system,
///    ensuring that, for example, a secondary body such as the Moon, is
///    NOT missing the acceleration of the Sun, IF the Earth either is
///    accelerating towards the Sun or is using ephemeris.
int main() {
    ///
    /// ENVIRONMENT SETUP
    ///
    entt::registry registry;

    // add termination condition for the at time 100 s
    auto t0 = 0.0;
    auto t_end = static_cast<double>(consts::JY) * 5;
    auto dt = static_cast<double>(consts::JD) / 24;

    const int N_BODIES = 3000;

    // physicsSystem
    auto print_system = PrintSystem();

    // create global reference frame
    auto global_frame = registry.create();
    registry.emplace<name>(global_frame, "global_frame");
    registry.emplace<position>(global_frame, 0., 0., 0.);
    registry.emplace<velocity>(global_frame, 0., 0., 0.);
    registry.emplace<acceleration>(global_frame, 0., 0., 0.);

    // SIMULATORS?
    const auto simulation_1 = registry.create();
    registry.emplace<name>(simulation_1, "simulation_1");
    registry.emplace<epoch>(simulation_1, t0);

    // termination
    const auto termination_1 = registry.create();
    registry.emplace<name>(termination_1, "time_termination");
    registry.emplace<termination>(termination_1, create_termination_condition(simulation_1, t_end));

    // add 100 random near earth asteroids
    for (int i = 0; i < N_BODIES; i++) {
        const auto body = registry.create();
        registry.emplace<name>(body, "BODY" + std::to_string(i + 1));
        // generate random position
        auto x = consts::AU * (1.0 * (rand() / static_cast<double>(RAND_MAX)) - 0.5);
        auto y = consts::AU * (1.0 * (rand() / static_cast<double>(RAND_MAX)) - 0.5);
        auto z = consts::AU * (1.0 * (rand() / static_cast<double>(RAND_MAX)) - 0.5);
        registry.emplace<position>(body, x, y, z);
        // generate random velocity
        auto vx = 27000 * (1.0 * (rand() / static_cast<double>(RAND_MAX)) - 0.5);
        auto vy = 27000 * (1.0 * (rand() / static_cast<double>(RAND_MAX)) - 0.5);
        auto vz = 27000 * (1.0 * (rand() / static_cast<double>(RAND_MAX)) - 0.5);
        registry.emplace<velocity>(body, vx, vy, vz);
        // add 0 acceleration
        registry.emplace<acceleration>(body, 0, 0, 0);
        // generate gravitational_parameter
        registry.emplace<gravitational_parameter>(body, consts::M_sun * consts::G / 1000);
        registry.emplace<simulated>(body, simulation_1);// flags as simulated
    }

    // add accelerations
    auto view_i = registry.view<name, position, velocity, gravitational_parameter>();
    auto view_j = registry.view<name, position, velocity, gravitational_parameter>();
    // iterate over all bodies
    for (auto body_i: view_i) {
        // iterate over all bodies again
        for (auto body_j: view_j) {
            // if the bodies are not the same
            if (body_i != body_j) {
                // add acceleration
                const auto acceleration = registry.create();
                std::cout << "Adding acceleration between " << view_i.get<name>(body_i) << " and " << view_j.get<name>(body_j) << std::endl;
                registry.emplace<name>(acceleration, "gravity_" + registry.get<name>(body_i) + "_" + registry.get<name>(body_j));
                registry.emplace<dynamic_influence>(acceleration, point_mass_acceleration(body_j, body_i));
            }
        }
    }
    // TODO: Add FLOPs counter, and FEVALs counter.

    bool terminate = false;

//#if ECSPACE_MATPLOTLIB
//    /// animation
//    auto animation_system = MatplotlibSystem(registry);// initialize the animation system
//                                                       //    animation_system.set_reference_frame(earth);
//                                                       //    animation_system.set_scale(consts::AU);
//    animation_system.initialize(registry);             // initialize the animation system
//#endif

    /// \note checks the dynamic system dependency components and marks
    /// all bodies with position and velocity being influenced by a
    /// dynamic influence.
    auto dynamics_system = DynamicsSystem(registry);

    // time the simulation
    auto start = std::chrono::high_resolution_clock::now();

    //    auto csv = CSVWriterSystem(registry);

    while (!terminate) {


        // get current time
        auto &e = registry.get<epoch>(simulation_1);

        //        std::cout << "e = " << registry.get<epoch>(simulation_1) / 60 / 60 / 24 << " days" << std::endl;
        //        std::tuple<double, std::string> t = get_significant_time_unit(registry.get<epoch>(simulation_1));
        std::cout << "t = " << prettify_time(e) << std::endl;
        //        get_significant_time_unit(registry.get<epoch>(simulation_1));

        // euler
        // Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        // Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry,  y, e);
        // dynamics_system.set_translational_state(registry, y + k1 * dt);

        // rk4
        Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        //        std::cout << "y = " << y.transpose() << std::endl;
        Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry, y, e);
        //        std::cout << "k1 = " << k1.transpose() << std::endl;

        Eigen::VectorXd k2 = dynamics_system.get_translational_state_derivative(registry, y + k1 * dt / 2, e + dt / 2);
        Eigen::VectorXd k3 = dynamics_system.get_translational_state_derivative(registry, y + k2 * dt / 2, e + dt / 2);
        Eigen::VectorXd k4 = dynamics_system.get_translational_state_derivative(registry, y + k3 * dt, e + dt);
        dynamics_system.set_translational_state(registry, y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6);

        //        csv.update(registry);

        // rk8
        // Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        // Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry,  y, e);
        // Eigen::VectorXd k2 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt / 3, e + dt / 3);
        // Eigen::VectorXd k3 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt / 12 + k2 * dt / 4, e + dt / 3);
        // Eigen::VectorXd k4 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt / 8 - k2 * dt / 2 + k3 * dt / 2, e + dt / 2);
        // Eigen::VectorXd k5 = dynamics_system.get_translational_state_derivative(registry,  y - k1 * dt / 2 + k2 * dt + k3 * dt * 2 - k4 * dt * 2, e + dt / 2);
        // Eigen::VectorXd k6 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt * 3 / 16 + k4 * dt * 9 / 16 + k5 * dt * 3 / 8, e + dt * 3 / 4);
        // Eigen::VectorXd k7 = dynamics_system.get_translational_state_derivative(registry,  y - k1 * dt * 3 / 7 + k2 * dt * 2 / 7 + k3 * dt * 12 / 7 - k4 * dt * 12 / 7 + k5 * dt * 8 / 7 - k6 * dt * 2 / 7, e + dt);
        // Eigen::VectorXd k8 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt * 7 / 90 + k3 * dt * 32 / 90 + k4 * dt * 12 / 90 + k5 * dt * 32 / 90 + k6 * dt * 7 / 90, e + dt);
        // dynamics_system.set_translational_state(registry, y + (k1 * 7 + k3 * 32 + k4 * 12 + k5 * 32 + k6 * 7) * dt / 90);

        // set new time
        registry.replace<epoch>(simulation_1, e + dt);

//#if ECSPACE_MATPLOTLIB
//        // animate
//        animation_system.update(registry);
//#endif

        // check if termination conditions are met
        auto view = registry.view<termination>();
        for (auto entity: view) {
            auto &term = view.get<termination>(entity);
            if (term(registry)) {
                std::cout << "Termination condition met: " << registry.get<name>(entity) << std::endl;
                registry.clear();
                // time the simulation
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "Elapsed time: " << elapsed.count() << " s " << std::endl;
                return 0;
            }
        }
    }
}