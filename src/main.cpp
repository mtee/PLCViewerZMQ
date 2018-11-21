#include <iostream>
#include <fstream>

#include "pointcloudmapping.h"

#include "opencv2/plot.hpp"


#include <zmq.hpp>
#include <Dealer.h>
#include <MessageProxy.h>
#include <attentive_framework.pb.h>

inline double toDouble(std::string s){
    std::replace(s.begin(), s.end(), ',', '.');
    return std::atof(s.c_str());
}


std::vector<std::string> split(const std::string& s, char delim) 
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    
    while (std::getline(ss, item, delim)) elems.push_back(item);
    
    return elems;
}

std::vector<std::string> split(const std::string& s) 
{
    std::istringstream iss(s);
    std::vector<std::string> elems;

    std::copy(
	std::istream_iterator<std::string>(iss),
	std::istream_iterator<std::string>(),
	std::back_inserter<std::vector<std::string>>(elems));

    return elems;
}


int main(int argc, const char* argv[])
{
    std::cout << "hello pointcloud" << std::endl;
    PointCloudMapping cloudMapper(0.02f);
    std::cout << "point cloud viewer created" << std::endl;
    cloudMapper.AddPointCloud("cloud.ply");


    std::ifstream posesFile("poses317.txt");
    if(posesFile.is_open()) {
        std::cout << "File opened" << std::endl;
        std::string line;
        std::vector<std::string> tokens;
        while (std::getline(posesFile, line))
        {
            if (line.length() < 1)
                continue;
            tokens = split(line);
            // Transformation
            cv::Mat pose;
            cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
            for(unsigned i = 0; i < 3; i++)
                for(unsigned j = 0; j < 4; j++)
                {
                    trans(i, j) =  toDouble(tokens[i*4 + j].c_str());
                }
            cloudMapper.AddOrUpdateFrustum("1", trans, 0.5, 1, 0, 0, 2);
            cv::waitKey(100);
        }
    }

    // press Enter in the console to quit
    std::cin.ignore();
}
