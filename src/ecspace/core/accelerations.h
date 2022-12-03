//
// Created by ggarrett on 11/30/22.
//

#ifndef ECSPACE_ACCELERATIONS_H
#define ECSPACE_ACCELERATIONS_H

#include "components.h"
#include "entt/entt.hpp"

dynamic_influence point_mass_acceleration(entt::entity entity_exerting, entt::entity entity_influenced) {
//#if ECSPACE_CUDA

//#else
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
//#endif
}



#endif//ECSPACE_ACCELERATIONS_H
