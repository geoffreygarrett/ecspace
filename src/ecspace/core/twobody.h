#ifndef ECSPACE_H
#define ECSPACE_H

/// \brief Sphere of influence calculation
/// \param r radial position of the minor body
/// \param M major body mass (or gravitational parameter)
/// \param m minor body mass (or gravitational parameter)
/// \return radius of the sphere of influence
double r_soi(double r, double m, double M) {
    return r * std::pow(m / M, 2.0 / 5.0);
}


struct orbital_elements {
    double a;
    double e;
    double i;
    double omega_p;
    double Omega;
    double nu;
};


#endif//ECSPACE_H