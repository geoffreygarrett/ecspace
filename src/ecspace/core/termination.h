//
// Created by ggarrett on 12/1/22.
//

#ifndef TUDAT_ENTT_TERMINATION_H
#define TUDAT_ENTT_TERMINATION_H

#include "components.h"
#include "entt/entt.hpp"
#include "twobody.h"
#include <functional>


typedef std::function<bool(entt::registry &)> termination;

termination create_termination_condition(entt::entity entity, double termination_time) {
    return [entity, termination_time](entt::registry &registry) {
        auto &time = registry.get<epoch>(entity);
        return time >= termination_time;
    };
}

termination create_exact_soi_exit_condition(entt::entity entity, entt::entity central_body, double scale_factor=1.0) {
    return [entity, central_body, scale_factor](entt::registry &registry) {
        auto &r_entity = registry.get<position>(entity);
        auto &r_central_body = registry.get<position>(central_body);
        auto &r_parent_central_body = registry.get<position>(registry.get<parent>(central_body));

        auto &mu_central_body = registry.get<gravitational_parameter>(central_body);
        auto &mu_parent_central_body = registry.get<gravitational_parameter>(registry.get<parent>(central_body));

        auto orbital_distance = std::sqrt(std::pow(r_entity.x - r_central_body.x, 2) +
                                           std::pow(r_entity.y - r_central_body.y, 2) +
                                           std::pow(r_entity.z - r_central_body.z, 2));

        auto orbital_distance_parent = std::sqrt(std::pow(r_central_body.x - r_parent_central_body.x, 2) +
                                                  std::pow(r_central_body.y - r_parent_central_body.y, 2) +
                                                  std::pow(r_central_body.z - r_parent_central_body.z, 2));

        // compute gravitational radius
        double gravitational_radius = r_soi(orbital_distance_parent, mu_central_body, mu_parent_central_body);

        // check if distance is larger than gravitational radius
        return orbital_distance > gravitational_radius * scale_factor;
    };
}


#endif//TUDAT_ENTT_TERMINATION_H
