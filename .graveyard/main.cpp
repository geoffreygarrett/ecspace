#include "entt/entt.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

#include "../src/ecspace/core/twobody.h"

#include "../src/ecspace/core/components.h"
#include "../src/ecspace/core/consts.h"
#include "../src/ecspace/core/systems.h"
#include "../src/ecspace/core/types.h"
#include "accelerations_old.h"

#include <Eigen/Dense>
#include <utility>

using namespace std;

//#include <boost/array.hpp>
//#include <boost/numeric/odeint.hpp>
//namespace odeint = boost::numeric::odeint;

using namespace acceleration_functions;


//class PatchConicsSystem {
//public:
//    PatchConicsSystem(entt::registry &registry) : registry(registry) {
//        // get all entities with a position and velocity
//        auto view = registry.view<position, velocity>();
//
//        // iterate over all entities with a position and velocity
//        for (auto entity : view) {
//            // get the position and velocity of the entity
//            auto &r = registry.get<position>(entity);
//            auto &v = registry.get<velocity>(entity);
//
//            // calculate the semi-major axis
//            double a = 1 / (2 / r.norm() - v.norm_squared() / consts::mu);
//
//            // calculate the eccentricity
//            double e = std::sqrt(1 - (r.cross(v).norm_squared() / (consts::mu * a)));
//
//            // calculate the inclination
//            double i = std::acos(r.z / r.norm());
//
//            // calculate the longitude of the ascending node
//            double omega = std::atan2(r.y, r.x);
//
//            // calculate the argument of periapsis
//            double omega_p = std::atan2(r.cross(v).z / (consts::mu * std::sin(i)), 1 - r.dot(v) * v.dot(v) / (consts::mu * a));
//
//            // calculate the true anomaly
//            double nu = std::atan2(r.dot(v) / (consts::mu * std::sqrt(a)), 1 - r.norm() / a);
//
//            // add the orbital elements to the entity
//            registry.emplace<orbital_elements>(entity, a, e, i, omega, omega_p, nu);
//        }
//    }
//
//    void dec
//};

//struct simulation {
//    entt::entity entity;
//    double t;
//    double dt;
//};

typedef entt::entity simulator;

//struct simulator {
//    simulation simulator;
//    double t;
//    double dt;
//};


struct euler_integrator {
    double t;
    double dt;
};


//struct epoch {
//    double current;
//    double previous;
//};


//struct relationship {
//    std::size_t children{};
//    entt::entity first{entt::null};
//    entt::entity prev{entt::null};
//    entt::entity next{entt::null};
//    entt::entity parent{entt::null};
//    // ... other data members ...
//};

typedef double independent;

//struct dynamic_influence {
//    entt::entity entity{entt::null};
//    entt::entity on_entity{entt::null};
//    const std::function<Eigen::Vector3d(entt::registry &, double)> function;
//    Eigen::Vector3d last_value;
//};

// class that constructs the acceleration functions for each entity based
// on the accelerations enabled for that entity, and the dependent entities
class AccelerationSystem {
public:
    void update(entt::registry &registry) {
        for (auto entity: registry.view<name>()) {
            acceleration total_acceleration = {0, 0, 0};
            for (auto &a: accelerations[entity]) {
                acceleration contributed_acceleration = a.get_acceleration(registry);
                total_acceleration.ddx += contributed_acceleration.ddx;
                total_acceleration.ddy += contributed_acceleration.ddy;
                total_acceleration.ddz += contributed_acceleration.ddz;
            }
            registry.emplace_or_replace<acceleration>(entity, total_acceleration);
        }
    }

    void add(entt::registry &registry, PointMassAcceleration acceleration) {
        accelerations[acceleration.get_entity_acted_upon()].push_back(acceleration);
    }


private:
    std::map<entt::entity, std::vector<PointMassAcceleration>> accelerations{};
};

class DynamicsSystem {
public:
    typedef int index;

    explicit DynamicsSystem(entt::registry &registry) {
        /// TODO: Perform checks that all entities have the required components
        // get count of all entities with a position and velocity
        //        auto view = registry.view<position, velocity, simulated>();
        //
        //        // iterate over all entities with a position and velocity and count
        //        for (auto entity : view) {
        //
        //            count_ += 1;
        //            std::cout<< "count: " << count_ << std::endl;
        //            // print name
        //            std::cout << "name: " << registry.get<name>(entity) << std::endl;
        //        }
    }

    void initialize(entt::registry &registry) {
        // get count of all entities with a position and velocity
        auto view = registry.view<position, velocity, simulated>();

        // iterate over all entities with a position and velocity and count
        for (auto entity: view) {
            // set index in to body
            registry.emplace_or_replace<index>(entity, count_);
            count_++;
        };

        // iteratre
    }

    Eigen::VectorXd get_translational_state(entt::registry &registry) {
        // get the name, position, velocity and acceleration of the whole system
        auto view = registry.view<position, velocity, simulated>();

        // TODO: is count effective?
        Eigen::VectorXd state = Eigen::VectorXd::Zero(6 * count_);

        // iterate over all entities with a name, position, velocity and acceleration,
        // with index i
        int i = 0;
        for (auto entity: view) {
            // get the position, velocity and acceleration of the entity
            auto &r = registry.get<position>(entity);
            auto &v = registry.get<velocity>(entity);

            // set the state vector
            state(6 * i) = r.x;
            state(6 * i + 1) = r.y;
            state(6 * i + 2) = r.z;
            state(6 * i + 3) = v.dx;
            state(6 * i + 4) = v.dy;
            state(6 * i + 5) = v.dz;
            i++;
        }

        return state;
    }

    void set_translational_state(entt::registry &registry, Eigen::VectorXd state) {
        // get the name, position, velocity and acceleration of the whole system
        auto view = registry.view<position, velocity, simulated>();

        int i = 0;

        // iterate over all entities with a name, position, velocity and acceleration
        for (auto entity: view) {
            // get the name, position, velocity and acceleration of the entity
            auto &name_ = registry.get<name>(entity);
            auto &position_ = registry.get<position>(entity);
            auto &velocity_ = registry.get<velocity>(entity);
            auto &acceleration_ = registry.get<acceleration>(entity);

            // set the state vector
            position_.x = state(6 * i);
            position_.y = state(6 * i + 1);
            position_.z = state(6 * i + 2);
            velocity_.dx = state(6 * i + 3);
            velocity_.dy = state(6 * i + 4);
            velocity_.dz = state(6 * i + 5);
            i++;
        }
    }

    Eigen::VectorXd get_translational_state_derivative(entt::registry &registry, entt::entity simulation, Eigen::VectorXd state, double t) {
        // get all entities with a name, position, velocity and acceleration
        //        auto view = registry.view<name, position, velocity, acceleration, simulated>();
        auto view = registry.view<const position, const velocity, const simulated, const index>();

        // store current state
        // TODO: If this is current state, this shouldn't be necessary, adds extra overhead
        Eigen::VectorXd translational_state_snapshot = get_translational_state(registry);

        // set provided state
        set_translational_state(registry, std::move(state));

        // create state derivative
        Eigen::VectorXd state_derivative = Eigen::VectorXd::Zero(6 * count_);

        //        int i = 0;

        // iterate over all entities with a name, position, velocity and acceleration
        //        std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&registry, &state_derivative, &i](auto entity) {
        //            auto &position_ = registry.get<position>(entity);
        //            auto &velocity_ = registry.get<velocity>(entity);
        //
        //            // set the state derivative vector
        //            state_derivative(6 * i) = velocity_.dx;
        //            state_derivative(6 * i + 1) = velocity_.dy;
        //            state_derivative(6 * i + 2) = velocity_.dz;
        //
        //            position_.x += velocity_.dx * dt;
        //            position_.y += velocity_.dy * dt;
        //            position_.z += velocity_.dz * dt;
        //
        //            i++;
        //        });

        std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&registry, &state_derivative, this](auto entity) {
            // get the name, position, velocity and acceleration of the entity
            auto &position_ = registry.get<position>(entity);
            auto &velocity_ = registry.get<velocity>(entity);
            auto &index_ = registry.get<index>(entity);

            // set the state derivative vector
            state_derivative(6 * index_) = velocity_.dx;
            state_derivative(6 * index_ + 1) = velocity_.dy;
            state_derivative(6 * index_ + 2) = velocity_.dz;
            acceleration total_acceleration = {0, 0, 0};

            for (auto &a: accelerations[entity]) {
                acceleration contributed_acceleration = a.get_acceleration(registry);
                total_acceleration.ddx += contributed_acceleration.ddx;
                total_acceleration.ddy += contributed_acceleration.ddy;
                total_acceleration.ddz += contributed_acceleration.ddz;
            }
            state_derivative(6 * index_ + 3) = total_acceleration.ddx;
            state_derivative(6 * index_ + 4) = total_acceleration.ddy;
            state_derivative(6 * index_ + 5) = total_acceleration.ddz;
        });

        // restore state
        set_translational_state(registry, translational_state_snapshot);
        return state_derivative;
    }

    //    void update(entt::registry &registry) {
    //        // get all entities with a name, position, velocity and acceleration
    //        auto view = registry.view<name, position, velocity, acceleration, simulated>();
    //
    //        // iterate over all entities with a name, position, velocity and acceleration
    //        for (auto entity: view) {
    //            // get the name, position, velocity and acceleration of the entity
    //            auto &name_ = registry.get<name>(entity);
    //            auto &position_ = registry.get<position>(entity);
    //            auto &velocity_ = registry.get<velocity>(entity);
    //            auto &acceleration_ = registry.get<acceleration>(entity);
    //
    //            // calculate the velocity
    //            velocity_.dx += acceleration_.ddx * dt;
    //            velocity_.dy += acceleration_.ddy * dt;
    //            velocity_.dz += acceleration_.ddz * dt;
    //
    //            // calculate the position
    //            position_.x += velocity_.dx * dt;
    //            position_.y += velocity_.dy * dt;
    //            position_.z += velocity_.dz * dt;
    //        }
    //    }

    void add(entt::registry &registry, PointMassAcceleration acceleration) {
        accelerations[acceleration.get_entity_acted_upon()].push_back(acceleration);
    }


private:
    std::map<entt::entity, std::vector<PointMassAcceleration>> accelerations{};
    int count_ = 0;
};


struct influences {
    std::vector<entt::entity> entities;
};

// TODO: Maybe add template based on size of registry.view<const dynamic_influence>();? Is this possible?
class DynamicsSystem2 {
public:
    typedef int index;

    explicit DynamicsSystem2(entt::registry &registry) {
        // assert that all entities with dynamic_influence point to an entity with position, velocity
        // TODO: Should this become an exception/  warning or be handled elsewhere with
        //   a loop which ensures that all entities with dynamic_influence point to an entity with position, velocity?
        //   Perhaps log a warning and attempt to retrieve from... i.e. spice.
        auto view = registry.view<const dynamic_influence>();
        auto count_ = 0;
        for (auto entity: view) {
            auto &dynamic_influence_ = registry.get<dynamic_influence>(entity);
            assert(registry.try_get<position>(dynamic_influence_.on_entity) != nullptr);
            assert(registry.try_get<velocity>(dynamic_influence_.on_entity) != nullptr);
            // TODO: Later we will need to consider rotational dynamics.
            // assign state_index and simulated
            registry.emplace_or_replace<index>(dynamic_influence_.on_entity, count_);
            registry.emplace_or_replace<simulated>(dynamic_influence_.on_entity);
            // TODO: Could form dependency graph for entities later... perhaps working towards co-simulation/ or better multi-threading!
            // add influences
            if (registry.try_get<influences>(dynamic_influence_.on_entity) == nullptr) {
                registry.emplace_or_replace<influences>(dynamic_influence_.on_entity);
            } else {
                registry.get<influences>(dynamic_influence_.on_entity).entities.push_back(entity);
            }
            count_++;
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

        // iterate over all simulated bodies
        std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&registry, t](auto entity) {
            // iterate over all dynamic_influences on this body and sum the accelerations
            auto &influences_ = registry.get<influences>(entity);
            auto index_ = registry.get<index>(entity);
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

        return state_derivative_;
    }

    void add(entt::registry &registry, PointMassAcceleration acceleration) {
        accelerations[acceleration.get_entity_acted_upon()].push_back(acceleration);
    }

private:
    std::map<entt::entity, std::vector<PointMassAcceleration>> accelerations{};
    int count_ = 0;
    Eigen::VectorXd state_;
    Eigen::VectorXd state_derivative_;
};

dynamic_influence point_mass_acceleration(entt::entity entity_exerting, entt::entity entity_influenced) {
    return dynamic_influence{entity_exerting, entity_influenced,
                             [=](entt::registry &registry, double t) -> Eigen::Vector3d {
                                 // get the position of the entity exerting the force
                                 auto &r1 = registry.get<position>(entity_exerting);

                                 // get the position of the entity acted upon
                                 auto &r2 = registry.get<position>(entity_influenced);

                                 // get the gravitational parameter of the entity exerting the force
                                 auto mu = registry.get<gravitational_parameter>(entity_exerting);

                                 // calculate the distance between the two entities
                                 double distance = std::sqrt(std::pow(r1.x - r2.x, 2) +
                                                             std::pow(r1.y - r2.y, 2) +
                                                             std::pow(r1.z - r2.z, 2));

                                 // calculate the acceleration
                                 return Eigen::Vector3d{
                                         -mu * (r2.x - r1.x) / std::pow(distance, 3),
                                         -mu * (r2.y - r1.y) / std::pow(distance, 3),
                                         -mu * (r2.z - r1.z) / std::pow(distance, 3)};
                             }};
}

struct butcher_tableau {
    Eigen::MatrixXd a;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    int s;
};

//struct rk4_butcher_tableau {
//    static butcher_tableau get() {
//        butcher_tableau tableau{};
//        tableau.a = Eigen::MatrixXd(4, 4);
//        tableau.a << 0, 0, 0, 0,
//                0.5, 0, 0, 0,
//                0, 0.5, 0, 0,
//                0, 0, 1,   0;
//        tableau.b = Eigen::VectorXd(4);
//        tableau.b << 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0;
//        tableau.c = Eigen::VectorXd(4);
//        tableau.c << 0, 0.5, 0.5, 1;
//        return tableau;
//    }
//};

butcher_tableau get_rk4_butcher_tableau() {
    Eigen::MatrixXd a(4, 4);
    a << 0, 0, 0, 0,
            0.5, 0, 0, 0,
            0, 0.5, 0, 0,
            0, 0, 1, 0;
    Eigen::VectorXd b(4);
    b << 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0;
    Eigen::VectorXd c(4);
    c << 0, 0.5, 0.5, 1;
    return butcher_tableau{a, b, c};
}

int main() {
    ///
    /// ENVIRONMENT SETUP
    ///
    entt::registry registry;

    // add termination condition for the at time 100 s
    auto t0 = 0.0;
    auto t_end = static_cast<double>(consts::JY) * 10;
    auto dt = static_cast<double>(consts::JD)/10;

    // physicsSystem
    auto print_system = PrintSystem();
    //    auto integration_system = PhysicsSystem(0.1);

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
    registry.emplace<euler_integrator>(simulation_1, 0.1);
    registry.emplace<epoch>(simulation_1, t0);

    // assign the simulated to the simulation_1 entity
    // TODO: Add check to ensure simulated { simulator } has the needed components.
    registry.emplace<simulated>(earth, simulation_1);
    registry.emplace<simulated>(moon, simulation_1);

    // termination
    typedef std::function<bool(entt::registry &)> termination;
    const auto time_termination = [&](entt::registry &registry) {
        auto e = registry.get<epoch>(simulation_1);
        return e >= t_end;
    };

    const auto exact_moon_exit_soi_termination = [&](entt::registry &registry) {
        auto &r_moon = registry.get<position>(moon);
        auto &r_earth = registry.get<position>(earth);
        auto &r_sun = registry.get<position>(sun);
        auto &mu_earth = registry.get<gravitational_parameter>(earth);
        auto &mu_sun = registry.get<gravitational_parameter>(sun);
        auto r_earth_norm = std::sqrt(std::pow(r_earth.x, 2) + std::pow(r_earth.y, 2) + std::pow(r_earth.z, 2));
        auto r_moon_norm = std::sqrt(std::pow(r_moon.x - r_earth.x, 2) + std::pow(r_moon.y - r_earth.y, 2) + std::pow(r_moon.z - r_earth.z, 2));
        auto r_earth_soi = r_soi(r_earth_norm, mu_earth, mu_sun);
        return r_moon_norm > r_earth_soi;
    };

    const auto termination_1 = registry.create();
    registry.emplace<name>(termination_1, "time_termination");
    registry.emplace<termination>(termination_1, time_termination);

    const auto termination_2 = registry.create();
    registry.emplace<name>(termination_2, "exact_moon_exit_soi_termination");
    registry.emplace<termination>(termination_2, exact_moon_exit_soi_termination);

    ///
    /// EQUATIONS OF MOTION
    ///
    /// NOTE: Must add a "on_delete/remove" hook to remove acceleration when the causal entity is deleted.
    ///       This is supported by EnTT, but I haven't implemented it yet.
    ///
    /// NOTES: Must add a check to ensure that the same acceleration function is not added twice for the same pair.
    ///

    const auto acceleration_1 = registry.create();
    registry.emplace<name>(acceleration_1, "sun_earth_gravity");
    registry.emplace<dynamic_influence>(acceleration_1, point_mass_acceleration(sun, earth));

    const auto acceleration_2 = registry.create();
    registry.emplace<name>(acceleration_2, "earth_moon_gravity");
    registry.emplace<dynamic_influence>(acceleration_2, point_mass_acceleration(earth, moon));

    const auto acceleration_3 = registry.create();
    registry.emplace<name>(acceleration_3, "sun_moon_gravity");
    registry.emplace<dynamic_influence>(acceleration_3, point_mass_acceleration(sun, moon));

    //    auto acceleration_system = AccelerationSystem();
    //    acceleration_system.add(registry, PointMassAcceleration(sun, earth));
    //    acceleration_system.add(registry, PointMassAcceleration(sun, moon));
    //    acceleration_system.add(registry, PointMassAcceleration(earth, moon));
    //    acceleration_system.add(registry, PointMassAcceleration(moon, earth));
    //
    //    auto dynamics_system = DynamicsSystem(registry);
    //    dynamics_system.add(registry, PointMassAcceleration(earth, moon));
    //    dynamics_system.add(registry, PointMassAcceleration(moon, earth));
    //    dynamics_system.add(registry, PointMassAcceleration(sun, earth));
    //    dynamics_system.add(registry, PointMassAcceleration(sun, moon));

    // add 100 random near earth asteroids
    int n_asteroids = 50;
    for (int i = 0; i < n_asteroids + 1; i++) {
        double ratio = static_cast<double>(i) / static_cast<double>(n_asteroids);
        const auto asteroid = registry.create();
        registry.emplace<name>(asteroid, "ASTEROID" + std::to_string(i));
        registry.emplace<position>(asteroid, 149598023000. - 384748000. - 384748000 / 10 * (10 * (0.1 + ratio * (1 - 0.5))), 0., 0.);
        registry.emplace<velocity>(asteroid, 0., 29780. + 1022. + 1000. / 100 * (10 * ratio - 0.5) - 10, 0.);
        registry.emplace<acceleration>(asteroid, 0, 0, 0);
        registry.emplace<gravitational_parameter>(asteroid, 4.902801e12);
        registry.emplace<parent>(asteroid, earth);
        registry.emplace<simulated>(asteroid, simulation_1);// flags as simulated

        const auto earth_asteroid_acceleration = registry.create();
        registry.emplace<name>(earth_asteroid_acceleration, "Acceleration_Earth_on_ASTEROID" + std::to_string(i));
        registry.emplace<dynamic_influence>(earth_asteroid_acceleration, point_mass_acceleration(earth, asteroid));

        const auto moon_asteroid_acceleration = registry.create();
        registry.emplace<name>(moon_asteroid_acceleration, "Acceleration_Moon_on_ASTEROID" + std::to_string(i));
        registry.emplace<dynamic_influence>(moon_asteroid_acceleration, point_mass_acceleration(moon, asteroid));

        const auto sun_asteroid_acceleration = registry.create();
        registry.emplace<name>(sun_asteroid_acceleration, "Acceleration_Sun_on_ASTEROID" + std::to_string(i));
        registry.emplace<dynamic_influence>(sun_asteroid_acceleration, point_mass_acceleration(sun, asteroid));

        //        dynamics_system.add(registry, PointMassAcceleration(earth, asteroid));
        //        dynamics_system.add(registry, PointMassAcceleration(moon, asteroid));
        //        dynamics_system.add(registry, PointMassAcceleration(sun, asteroid));
        //
        //        acceleration_system.add(registry, PointMassAcceleration(earth, asteroid));
        //        acceleration_system.add(registry, PointMassAcceleration(moon, asteroid));
        //        acceleration_system.add(registry, PointMassAcceleration(sun, asteroid));
    }


    bool terminate = false;

    /// csv
    //    auto csv_system = CSVWriterSystem(registry);

    /// animation
    auto animation_system = AnimationSystem(registry);// initialize the animation system
    animation_system.set_reference_frame(earth);
    animation_system.set_scale(consts::AU);
    animation_system.initialize(registry);// initialize the animation system

    // dynamics
    //    dynamics_system.initialize(registry);

    //    auto physics_system = PhysicsSystem();

    /// \note checks the dynamic system dependency components and marks
    /// all bodies with position and velocity being influenced by a
    /// dynamic influence.
    auto dynamics_system = DynamicsSystem2(registry);

    // time the simulation
    auto start = std::chrono::high_resolution_clock::now();

    while (!terminate) {

        // get current time

        auto &e = registry.get<epoch>(simulation_1);

        // euler
        Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry,  y, e);
        dynamics_system.set_translational_state(registry, y + k1 * dt);

        // rk4
//        Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
//        Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry,  y, e);
//        Eigen::VectorXd k2 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt / 2, e + dt / 2);
//        Eigen::VectorXd k3 = dynamics_system.get_translational_state_derivative(registry,  y + k2 * dt / 2, e + dt / 2);
//        Eigen::VectorXd k4 = dynamics_system.get_translational_state_derivative(registry,  y + k3 * dt, e + dt);
//        dynamics_system.set_translational_state(registry, y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6);


        // rk4
        //        butcher_tableau rk4_butcher_tableau = get_rk4_butcher_tableau();
        //        auto a = rk4_butcher_tableau.a;
        //        auto b = rk4_butcher_tableau.b;
        //        auto c = rk4_butcher_tableau.c;
        //        auto s = rk4_butcher_tableau.s;
        //        auto temp_e = e;

        // rk8
//        Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
//        Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry,  y, e);
//        Eigen::VectorXd k2 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt / 3, e + dt / 3);
//        Eigen::VectorXd k3 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt / 12 + k2 * dt / 4, e + dt / 3);
//        Eigen::VectorXd k4 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt / 8 - k2 * dt / 2 + k3 * dt / 2, e + dt / 2);
//        Eigen::VectorXd k5 = dynamics_system.get_translational_state_derivative(registry,  y - k1 * dt / 2 + k2 * dt + k3 * dt * 2 - k4 * dt * 2, e + dt / 2);
//        Eigen::VectorXd k6 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt * 3 / 16 + k4 * dt * 9 / 16 + k5 * dt * 3 / 8, e + dt * 3 / 4);
//        Eigen::VectorXd k7 = dynamics_system.get_translational_state_derivative(registry,  y - k1 * dt * 3 / 7 + k2 * dt * 2 / 7 + k3 * dt * 12 / 7 - k4 * dt * 12 / 7 + k5 * dt * 8 / 7 - k6 * dt * 2 / 7, e + dt);
//        Eigen::VectorXd k8 = dynamics_system.get_translational_state_derivative(registry,  y + k1 * dt * 7 / 90 + k3 * dt * 32 / 90 + k4 * dt * 12 / 90 + k5 * dt * 32 / 90 + k6 * dt * 7 / 90, e + dt);
//        dynamics_system.set_translational_state(registry, y + (k1 * 7 + k3 * 32 + k4 * 12 + k5 * 32 + k6 * 7) * dt / 90);


        // Two-step Adams–Bashforth
        //        Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        //        Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry, simulation_1, y, e);
        //        Eigen::VectorXd k2 = dynamics_system.get_translational_state_derivative(registry, simulation_1, y + k1 * dt, e + dt);
        //        dynamics_system.set_translational_state(registry, y + (k1 + k2) * dt / 2);


        // euler
        //        Eigen::VectorXd y = dynamics_system.get_translational_state(registry);
        //        Eigen::VectorXd k1 = dynamics_system.get_translational_state_derivative(registry, simulation_1, y, e);
        //        dynamics_system.set_translational_state(registry, y + k1 * dt);
        //        dynamics_system.update(registry);

        // set new time
        registry.replace<epoch>(simulation_1, e + dt);

        // animate
        animation_system.update(registry);

        // check if termination conditions are met
        auto view = registry.view<termination>();
        for (auto entity: view) {
            auto &term = view.get<termination>(entity);
            if (term(registry)) {
                std::cout << "Termination condition met: " << registry.get<name>(entity) << std::endl;
                terminate = true;
                break;
            }
        }

        if (terminate) {
            // destroy all
            registry.clear();
            break;
        };
        //        csv_system.update(registry);
    }
    // time the simulation
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s " << std::endl;
    // time wait
    //    update(registry);
}