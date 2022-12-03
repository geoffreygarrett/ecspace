//
// Created by ggarrett on 11/30/22.
//

#ifndef TUDAT_ENTT_CONSTS_H
#define TUDAT_ENTT_CONSTS_H

namespace consts {
    ///
    /// \brief Gravitational Constant (a.k.a. Cavendish Constant)
    /// \details The gravitational constant is a physical constant that relates the strength of
    /// Newtonian gravity to the masses of objects. It is usually denoted by the letter G.
    ///
    const double G = 6.67408e-11;
    ///
    /// \brief Speed of light [m/s]
    /// \details https://en.wikipedia.org/wiki/Speed_of_light
    ///
    const double c = 299792458;
    ///
    /// \brief Astronomical Unit [AU]
    /// \details https://en.wikipedia.org/wiki/Astronomical_unit
    ///
    const double AU = 149597870700;
    ///
    /// \brief Mass of the Sun [kg]
    /// \details https://en.wikipedia.org/wiki/Solar_mass
    ///
    const double M_sun = 1.98855e30;
    ///
    /// \brief Mass of the Earth [kg]
    /// \details https://en.wikipedia.org/wiki/Earth_mass
    ///
    const double M_earth = 5.97237e24;
    ///
    /// \brief Mass of the Moon [kg]
    ///
    const double M_moon = 7.34767309e22;
    ///
    /// \brief Mass of Mars [kg]
    ///
    const double M_mars = 6.4171e23;
    ///
    /// \brief Mass of Jupiter [kg]
    ///
    const double M_jupiter = 1.8986e27;
    ///
    /// \brief Mass of Saturn [kg]
    ///
    const double M_saturn = 5.6834e26;
    ///
    /// \brief Mass of Uranus [kg]
    ///
    const double M_uranus = 8.6810e25;
    ///
    /// \brief Mass of Neptune [kg]
    ///
    const double M_neptune = 1.0243e26;
    ///
    /// \brief Mass of Pluto [kg]
    ///
    const double M_pluto = 1.303e22;
    ///
    /// \brief J2000 epoch [s]
    ///
    const double J2000 = 2451545.0;
    ///
    /// \brief Julian day [s]
    ///
    const double JD = 86400.0;
    ///
    /// \brief Julian year [s]
    ///
    const double JY = 365.25 * JD;
    ///
    /// \brief Julian century [s]
    ///
    const double JC = 100 * JY;

}// namespace consts

#endif//TUDAT_ENTT_CONSTS_H
