//
// Created by ggarrett on 11/30/22.
//

#ifndef TUDAT_ENTT_ACCELERATIONS_OLD_H
#define TUDAT_ENTT_ACCELERATIONS_OLD_H

#include "../src/ecspace/core/components.h"
#include "entt/entt.hpp"


namespace acceleration_functions {

    // typedef for lambda returning acceleration
    typedef std::function<acceleration(entt::registry &)> AccelerationFunction;

    class AccelerationBase {
    public:
        virtual acceleration get_acceleration(entt::registry &registry) = 0;
    };
    ///
    /// \brief acceleration struct
    ///
    class Acceleration : public AccelerationBase {
    public:
        Acceleration(entt::entity entity_exerting_force, entt::entity entity_acted_upon)
            : entity_exerting_force_(entity_exerting_force), entity_acted_upon_(entity_acted_upon){};

        acceleration get_acceleration(entt::registry &registry) override {
            return acceleration{0, 0, 0};
        }

        [[maybe_unused]] entt::entity get_entity_exerting_force() { return entity_exerting_force_; };

        [[maybe_unused]] entt::entity get_entity_acted_upon() { return entity_acted_upon_; };

    protected:
        entt::entity entity_exerting_force_{entt::null};
        entt::entity entity_acted_upon_{entt::null};

    };


    class PointMassAcceleration : public Acceleration {
    public:
        PointMassAcceleration(entt::entity entity_exerting_force, entt::entity entity_acted_upon)
            : Acceleration(entity_exerting_force, entity_acted_upon){};

        acceleration get_acceleration(entt::registry &registry) final {
            // get the position of the entity exerting the force
            auto &r1 = registry.get<position>(entity_exerting_force_);

            // get the position of the entity acted upon
            auto &r2 = registry.get<position>(entity_acted_upon_);

            // get the gravitational parameter of the entity exerting the force
            auto mu = registry.get<gravitational_parameter>(entity_exerting_force_);

            // if entity experiencing has a gravitational parameter, sum it with mu
            //            auto mu2 = registry.try_get<gravitational_parameter>(entity_acted_upon_);

            //            // if not null ptr, add it to mu
            //            if (mu2 != nullptr) mu += *mu2;
            //                // output bodyname
            ////                std::cout << registry.get<name>(entity_acted_upon_) << std::endl;
            ////                std::cout<< "adding gravitational parameter" << std::endl;
            ////                std::cout<< "mu2: " << *mu2 << std::endl;
            //
            //                // update mu


            // calculate the distance between the two entities
            double distance = std::sqrt(std::pow(r1.x - r2.x, 2) +
                                        std::pow(r1.y - r2.y, 2) +
                                        std::pow(r1.z - r2.z, 2));

            // calculate the acceleration
            acceleration acceleration{};
            acceleration.ddx = mu * (r1.x - r2.x) / std::pow(distance, 3);
            acceleration.ddy = mu * (r1.y - r2.y) / std::pow(distance, 3);
            acceleration.ddz = mu * (r1.z - r2.z) / std::pow(distance, 3);
            return acceleration;
        };
    };

    AccelerationFunction radiation_pressure(
            entt::registry &registry,
            entt::entity entity_exerting,
            entt::entity entity_experiencing) {

        return [=](entt::registry &registry) {
            auto &r1 = registry.get<position>(entity_exerting);
            auto &r2 = registry.get<position>(entity_experiencing);

            // calculate distance between bodies
            double r = sqrt(pow(r1.x - r2.x, 2) + pow(r1.y - r2.y, 2) + pow(r1.z - r2.z, 2));

            // calculate acceleration
            acceleration a{};
            a.ddx = 0;
            a.ddy = 0;
            a.ddz = 0;
            return a;
        };
    }

}// namespace acceleration_functions


#endif//TUDAT_ENTT_ACCELERATIONS_OLD_H
