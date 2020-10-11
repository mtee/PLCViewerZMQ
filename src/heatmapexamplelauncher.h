#ifndef HEATMAPEXAMPLELAUNCHER_H
#define HEATMAPEXAMPLELAUNCHER_H

#include <iostream>
#include <fstream>

#include "pointcloudmapping.h"
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>

class HeatMapExampleLauncher
{
    public:
        HeatMapExampleLauncher();

        static int runCockpitExample(bool useSemanticGrid = false);
        static int runPCAOfficeExample(bool useSemanticGrid = false);
        static int runVoestExample(bool useSemanticGrid = false);

    private:
        static double toDouble(std::string s);
        static std::vector<std::string> split(const std::string& s, char delim);
        static std::vector<std::string> split(const std::string& s);
        static int loadPointCloudFromBIN(std::string filename, PointCloudMapping& cloudMapper, uint32_t nb_clusters);
};

#endif