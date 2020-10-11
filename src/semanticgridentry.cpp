
#include "semanticgridentry.h"

SemanticGridEntry::SemanticGridEntry(std::string _id, std::string _name, Eigen::Vector3f _minCorner, Eigen::Vector3f _maxCorner) : idTag(_id),nameTag(_name),minCorner(_minCorner),maxCorner(_maxCorner) {
}

void SemanticGridEntry::increment(){
    heatMapValue++;
}

void SemanticGridEntry::setAlreadyIncremented(bool flag){
    incrFlag = flag;
}

bool SemanticGridEntry::getAlreadyIncremented(){
    return incrFlag;
}

int SemanticGridEntry::getHeatMapValue(){
    return heatMapValue;
}
