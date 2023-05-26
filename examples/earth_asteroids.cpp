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


#include <cuda.h>
#include <cuda_runtime.h>
#include "../src/ecspace/core/test.cuh"



//#include "vector_types.h"

/// \important
/// - TODO: Employ a sanity check for all accelerations in the system,
///    ensuring that, for example, a secondary body such as the Moon, is
///    NOT missing the acceleration of the Sun, IF the Earth either is
///    accelerating towards the Sun or is using ephemeris.

//__host__ __device__
//void point_mass_acceleration(double GM, float4 r1, float3 r2){
//    printf("%s, %d %d\n", s, i, k);
//}

int main() {
    ///
    /// ENVIRONMENT SETUP
    ///
    entt::registry registry;

    // add termination condition for the at time 100 s
    auto t0 = 0.0;
    auto t_end = static_cast<double>(consts::JY) * 5;
    auto dt = static_cast<double>(consts::JD) / 24 * 6;

    // create global reference frame
    auto global_frame = registry.create();
    registry.emplace<name>(global_frame, "global_frame");
    registry.emplace<position>(global_frame, 0., 0., 0.);
    registry.emplace<velocity>(global_frame, 0., 0., 0.);
    registry.emplace<acceleration>(global_frame, 0., 0., 0.);

    // create sun
    auto sun = registry.create();
    registry.emplace<name>(sun, "Sun");
    registry.emplace<position>(sun, 0, 0, 0);
    registry.emplace<velocity>(sun, 0, 0, 0);
    registry.emplace<acceleration>(sun, 0, 0, 0);
    registry.emplace<gravitational_parameter>(sun, 1.32712440042e20);
    registry.emplace<parent>(sun, global_frame);

    // create a celestial body
    const auto earth = registry.create();
    registry.emplace<name>(earth, "Earth");
    registry.emplace<position>(earth, 149598023000., 0., 0.);
    registry.emplace<velocity>(earth, 0., 29780., 0.);
    registry.emplace<gravitational_parameter>(earth, 3.986004418e14);
    registry.emplace<parent>(earth, sun);
    registry.emplace<acceleration>(earth, 0, 0, 0);

    // create another celestial body
    const auto moon = registry.create();
    registry.emplace<name>(moon, "Moon");
    registry.emplace<position>(moon, 149598023000. + 384748000, 0., 0.);
    registry.emplace<velocity>(moon, 0., 29780. + 1022., 0.);
    registry.emplace<gravitational_parameter>(moon, 4.902801e12);
    registry.emplace<parent>(moon, earth);
    registry.emplace<acceleration>(moon, 0, 0, 0);

    // SIMULATORS?
    const auto simulation_1 = registry.create();
    registry.emplace<name>(simulation_1, "simulation_1");
    registry.emplace<epoch>(simulation_1, t0);

//    entt::sigh<bool()> terminate_signal;
//    entt::sink terminate_sink{terminate_signal};

    // termination
    const auto termination_1 = registry.create();
    registry.emplace<name>(termination_1, "time_termination");
    registry.emplace<termination>(termination_1, create_termination_condition(simulation_1, t_end));

    const auto termination_2 = registry.create();
    registry.emplace<name>(termination_2, "exact_moon_exit_soi_termination");
    registry.emplace<termination>(termination_2, create_exact_soi_exit_condition(moon, earth));

    //
    // EQUATIONS OF MOTION
    //
    // NOTE: Must add a "on_delete/remove" hook to remove acceleration when the causal entity is deleted.
    //       This is supported by EnTT, but I haven't implemented it yet.
    //
    // NOTES: Must add a check to ensure that the same acceleration function is not added twice for the same pair.
    //
    /// \ACCELERATIONS
    const auto acceleration_1 = registry.create();
    registry.emplace<name>(acceleration_1, "sun_earth_gravity");
    registry.emplace<dynamic_influence>(acceleration_1, point_mass_acceleration(sun, earth));

    const auto acceleration_2 = registry.create();
    registry.emplace<name>(acceleration_2, "earth_moon_gravity");
    registry.emplace<dynamic_influence>(acceleration_2, point_mass_acceleration(earth, moon));

    const auto acceleration_3 = registry.create();
    registry.emplace<name>(acceleration_3, "sun_moon_gravity");
    registry.emplace<dynamic_influence>(acceleration_3, point_mass_acceleration(sun, moon));

    // add 100 random near earth asteroids
    int n_asteroids = 20;
    for (int i = 0; i < n_asteroids + 1; i++) {
        double ratio = static_cast<double>(i) / static_cast<double>(n_asteroids);
        const auto asteroid = registry.create();
        registry.emplace<name>(asteroid, "ASTEROID" + std::to_string(i));
        registry.emplace<position>(asteroid, 149598023000. - 384748000. - 384748000 / 10 * (10 * (0.1 + ratio * (1 - 0.5))), 0., 0.);
        registry.emplace<velocity>(asteroid, 0., 29780. + 1022, 0.);
        registry.emplace<acceleration>(asteroid, 0, 0, 0);
        registry.emplace<gravitational_parameter>(asteroid, 4.902801e12);
        registry.emplace<parent>(asteroid, earth);
        registry.emplace<simulated>(asteroid, simulation_1);// flags as simulated

        /// \TERMINATION
        const auto termination_asteroid = registry.create();
        registry.emplace<name>(termination_asteroid, "exact_ASTEROID" + std::to_string(i) + "_exit_soi_termination");
        registry.emplace<termination>(termination_asteroid, create_exact_soi_exit_condition(asteroid, earth, 3.0));

        /// \ACCELERATIONS
        const auto earth_asteroid_acceleration = registry.create();
        registry.emplace<name>(earth_asteroid_acceleration, "Acceleration_Earth_on_ASTEROID" + std::to_string(i));
        registry.emplace<dynamic_influence>(earth_asteroid_acceleration, point_mass_acceleration(earth, asteroid));

        const auto moon_asteroid_acceleration = registry.create();
        registry.emplace<name>(moon_asteroid_acceleration, "Acceleration_Moon_on_ASTEROID" + std::to_string(i));
        registry.emplace<dynamic_influence>(moon_asteroid_acceleration, point_mass_acceleration(moon, asteroid));

        const auto sun_asteroid_acceleration = registry.create();
        registry.emplace<name>(sun_asteroid_acceleration, "Acceleration_Sun_on_ASTEROID" + std::to_string(i));
        registry.emplace<dynamic_influence>(sun_asteroid_acceleration, point_mass_acceleration(sun, asteroid));
    }
    // TODO: Add FLOPs counter, and FEVALs counter.

    bool terminate = false;


    /// \note checks the dynamic system dependency components and marks
    /// all bodies with position and velocity being influenced by a
    /// dynamic influence.
    auto dynamics_system = DynamicsSystem(registry);

    // time the simulation
    auto start = std::chrono::high_resolution_clock::now();

    //    auto csv = CSVWriterSystem(registry);

//    int f() { return 0; }
//    int g() { return 1; }

    // ...
//    float *d_buf;
//    float *d_buf;
//
//    cudaMalloc(&d_buf, 100 * sizeof(float));
//    cudaMalloc(&d_buf, sizeof(position) * (n_asteroids + 2));

//    position *d_buf;
//    cudaMalloc(&d_buf, sizeof(position) * (n_asteroids + 2));
    device_info();

    while (!terminate) {
//        device_info();

//        wrapper();
        // get current time
        auto &e = registry.get<epoch>(simulation_1);
        std::cout << "t = " << prettify_time(registry.get<epoch>(simulation_1)) << std::endl;

        // euler
        // Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        // Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry,  y, e);
        // dynamics_system.set_translational_state(registry, y + k1 * dt);

        // rk4
        Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry, y, e);
        Eigen::VectorXd k2 = dynamics_system.get_translational_state_derivative(registry, y + k1 * dt / 2, e + dt / 2);
        Eigen::VectorXd k3 = dynamics_system.get_translational_state_derivative(registry, y + k2 * dt / 2, e + dt / 2);
        Eigen::VectorXd k4 = dynamics_system.get_translational_state_derivative(registry, y + k3 * dt, e + dt);
        dynamics_system.set_translational_state(registry, y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6);



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

        //        // check if termination conditions are met
        auto view = registry.view<termination>();
        for (auto entity: view) {
            auto &term = view.get<termination>(entity);
            if (term(registry)) {
                std::cout << "Termination condition met: " << registry.get<name>(entity) << std::endl;
                terminate = true;
            }
        }
    }
    // time the simulation
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s " << std::endl;
}