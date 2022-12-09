//
// Created by ggarrett on 12/9/22.
//

#ifndef ECSPACE_HELPER_H
#define ECSPACE_HELPER_H



std::string prettify_time(double time){
    // uses all time units up to most significant
    // time unit that is not 0
    std::string time_string;
    double time_left = time;
    int years = time_left / 365.25 / 24 / 60 / 60;
    time_left -= years * 365.25 * 24 * 60 * 60;
    int days = time_left / 24 / 60 / 60;
    time_left -= days * 24 * 60 * 60;
    int hours = time_left / 60 / 60;
    time_left -= hours * 60 * 60;
    int minutes = time_left / 60;
    time_left -= minutes * 60;
    double seconds = time_left;
    if (years > 0) {
        time_string += std::to_string(years) + " years ";
    }
    if (days > 0) {
        time_string += std::to_string(days) + " days ";
    }
    if (hours > 0) {
        time_string += std::to_string(hours) + " hours ";
    }
    if (minutes > 0) {
        time_string += std::to_string(minutes) + " minutes ";
    }
    if (seconds > 0) {
        time_string += std::to_string(seconds) + " seconds ";
    }
    return time_string;


}


#endif//ECSPACE_HELPER_H
