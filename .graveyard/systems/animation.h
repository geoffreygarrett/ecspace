//
// Created by ggarrett on 12/1/22.
//

#ifndef SPACENTITY_ANIMATION_H
#define SPACENTITY_ANIMATION_H

//#include <opencv2/opencv.hpp>
//#include <opencv

class OpenCVSystem {
public:
    explicit OpenCVSystem(entt::registry &registry) {
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
            plt::plot(soi_x, soi_y, "k--");
            plt::legend();
            plt::pause(0.000001);
        }
        i++;
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


#endif//SPACENTITY_ANIMATION_H
