//
//
//
#ifndef TUDAT_ENTT_COMPONENTS_H
#define TUDAT_ENTT_COMPONENTS_H

#include <Eigen/Dense>

// eigen
typedef double epoch;
typedef std::string name;
typedef double gravitational_parameter;
typedef double mass;

typedef entt::entity simulator;

//struct relationship {
//    std::size_t children{};
//    entt::entity first{entt::null};
//    entt::entity prev{entt::null};
//    entt::entity next{entt::null};
//    entt::entity parent{entt::null};
//    // ... other data members ...
//};

typedef double independent;






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


struct dynamic_influence {
    entt::entity entity{entt::null};
    entt::entity on_entity{entt::null};
    const std::function<Eigen::Vector3d(entt::registry &, double)> function;
    Eigen::Vector3d last_value;
};

struct influences {std::vector<entt::entity> entities;};

typedef entt::entity parent;

struct simulated {entt::entity simulator;};

struct position {
    double x;
    double y;
    double z;
};



struct velocity {
    double dx;
    double dy;
    double dz;
};

struct acceleration {
    double ddx;
    double ddy;
    double ddz;
};

struct ephemeris {
    std::vector<epoch> times;
    std::vector<position> positions;
    std::vector<velocity> velocities;
    std::vector<acceleration> accelerations;
};

#endif//TUDAT_ENTT_COMPONENTS_H
