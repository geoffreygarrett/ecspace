//
// Created by ggarrett on 12/1/22.
//

#include "../vendor/cv-plot/CvPlot/inc/CvPlot/cvplot.h"

int main () {

    std::vector<double> x(20*1000), y1(x.size()), y2(x.size()), y3(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = i * CV_2PI / x.size();
        y1[i] = std::sin(x[i]);
        y2[i] = y1[i] * std::sin(x[i]*50);
        y3[i] = y2[i] * std::sin(x[i]*500);
    }
    auto axes = CvPlot::makePlotAxes();
    axes.create<CvPlot::Series>(x, y3, "-g");
    axes.create<CvPlot::Series>(x, y2, "-b");
    axes.create<CvPlot::Series>(x, y1, "-r");

    //plot to a cv::Mat
    cv::Mat mat = axes.render(300, 400);

    //or show with interactive viewer
    CvPlot::show("mywindow", axes);
}