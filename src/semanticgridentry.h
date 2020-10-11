#ifndef SEMANTICGRIDENTRY_H
#define SEMANTICGRIDENTRY_H

#include <vector>

#include "Eigen/Core"

class SemanticGridEntry
{
    public:
    SemanticGridEntry(std::string _id, std::string _name, Eigen::Vector3f _minCorner, Eigen::Vector3f _maxCorner);

    void increment();
    void setAlreadyIncremented(bool flag);
    bool getAlreadyIncremented();
    int getHeatMapValue();

    const std::string idTag;
    const std::string nameTag;
    
    const Eigen::Vector3f minCorner;
    const Eigen::Vector3f maxCorner;

    private:
    int heatMapValue = 0;
    bool incrFlag = false;


};

#endif // SEMANTICGRIDENTRY_H
