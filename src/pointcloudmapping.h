/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include <map>
#include <vector>
#include <sstream>
#include <condition_variable>
#include <assert.h>
#include <math.h>

#include "semanticgridentry.h"

//#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//#include <opencv2/xfeatures2d.hpp>

#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud_adjacency.h>

#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/common/centroid.h>

#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace boost;

class PointCloudMapping
{


public:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef double coord1_t; // one dimension
    typedef cv::Vec<coord1_t, 3> coord3_t; // three dimensions
    typedef pcl::visualization::MouseEvent::Type MouseType;
    typedef pcl::visualization::MouseEvent::MouseButton MouseButton;

    // rgb-d
    typedef cv::Vec<uchar, 3> bgr_t;
    typedef double depth_t;

    // image types
    typedef cv::Mat_<coord3_t> img_coord_t;
    typedef cv::Mat_<bgr_t> img_bgr_t;
    typedef cv::Mat_<depth_t> img_depth_t;

    std::map<std::string, cv::Mat> _frustums;
    std::map<std::string,cv::Mat>::iterator frustumIterator;

    PointCloudMapping( double resolution_, int _windowHeight = 1080, int _windowWidth = 1920);
    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* junk);
    void mouseEventOccurred (const pcl::visualization::MouseEvent &event, void* junk);
    void AddPointCloud(std::string filename);
    void AddPointCloud(cv::Mat transform, cv::Mat color, img_coord_t& _cloud, std::string frustumId);
    void AddPointCloud(std::vector<cv::Vec3f> pointList, std::vector<cv::Vec3b> colorsList,float scale = 1);
    void AddPointCloudFromOBJ(std::string filename);
    void AddTexturedPolygonFromOBJ(std::string filename);
    void generateSearchOctree(float res, int _nPointsForSolid);
    bool addToSemanticGrid(std::string id, std::string name, Eigen::Vector3f minCorner, Eigen::Vector3f maxCorner);
    bool addToSemanticGrid(std::string id, std::string name, float xcenter, float ycenter, float zcenter, float xsize, float ysize, float zsize);
    void addTitlesToSemanticGrid();
    bool readSemanticGridFromJSON(std::string filename);
    void AddTrainingFrameToPointCloud(cv::Mat& camMatrix, cv::Mat transform, cv::Mat color, cv::Mat depth, std::string frustumId);
    void AddMeshToPointCloud(cv::Mat& camMatrix, cv::Mat transform, cv::Mat color, cv::Mat depth, std::string frustumId);
    cv::Mat AddTestFrameToPointCloud(cv::Mat& camMatrix, cv::Mat transform, cv::Mat color, cv::Mat depth, img_coord_t& cloud, cv::Mat_<cv::Point2i>& sampling, std::string frustumId);
    void Add2D3DCorrespondencesToPointCloud(std::vector< float > c3D);
    PointCloud::Ptr extractIndices(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud, const pcl::IndicesPtr & indices, bool negative, bool keepOrganized);
    void mouseMoveEvent(const pcl::visualization::MouseEvent &event, void* junk);
    void Shutdown();
    void Visualize();
    void OptimizePointCloud();
    void RemoveFrustum(const std::string & id);
    void AddOrUpdateFrustum(const std::string & id, const cv::Mat & cvtransform, float scale, double r, double g, double b, float lineWidth, bool useOctree, cv::Mat & rotCompensation, float gazeNormX = 0., float gazeNormY = 0., bool isCockpitCase = false);
    void AddOrUpdateFrustum(const std::string & id, const cv::Mat & cvtransform, float scale, double r, double g, double b, float lineWidth, cv::Mat & rotCompensation);
    void pointPickingEventOccurred (const pcl::visualization::PointPickingEvent& event, void* viewer_void);
    pcl::IndicesPtr radiusFiltering(pcl::PointCloud<PointT>::Ptr & cloud, const pcl::IndicesPtr & indices, float radiusSearch, int minNeighborsInRadius);
    cv::Vec3f GetCloudCentroid();
    void setSemanticMode(bool val);

private:
    PointCloud::Ptr generatePointCloud(float cx, float cy, float fx, float fy, cv::Mat transform, cv::Mat color, cv::Mat depth, bool gtData, int decimateDepth, float depthMax = 10.0);
    std::vector<pcl::Vertices> getOrganizedFastMesh(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud, double angleTolerance, int trianglePixelSize, const Eigen::Vector3f & viewpoint);
    std::vector<int> filterNotUsedVerticesFromMesh(const pcl::PointCloud<pcl::PointXYZRGB> & cloud, const std::vector<pcl::Vertices> & polygons, pcl::PointCloud<pcl::PointXYZRGB> & outputCloud, std::vector<pcl::Vertices> & outputPolygons);
    bool addTextureMesh(shared_ptr<pcl::visualization::PCLVisualizer> _visualizer,  pcl::TextureMesh::Ptr mesh, const cv::Mat & image, const std::string &id, int viewport);
    void voxelFilterCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    void updateSemanticGridLabels();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelize(
            const typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
            const pcl::IndicesPtr & indices,
            float voxelSize);
    PointCloud::Ptr globalMap;
    PointCloud::Ptr displayCloud;

    Eigen::Vector3i getVoxelIndex(Eigen::Vector3f pos);
    Eigen::Vector3i getVoxelIndex(std::string indexStr);
    std::string getVoxelIndexString(Eigen::Vector3f pos);
    std::string getVoxelIndexString(Eigen::Vector3i indexVec);
    Eigen::Vector3f getVoxelCenter(std::string indexStr);
    Eigen::Vector3f getVoxelCenter(Eigen::Vector3i indexVec);    
    Eigen::Vector3f getColourFromValue(double v, double vmin, double vmax);
    Eigen::Vector3f getCornersFromCenterPlusSize();

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::Ptr octreeSearch;
    double voxelXmin, voxelYmin, voxelZmin, voxelXmax, voxelYmax, voxelZmax;
    double voxelSideLength;
    int nViews, nPointsForSolid;
    std::string lastMapMax;

    std::vector<std::string> activeCubeIndices;
    std::map<std::string, int> attentionHeatMap;

    bool gotPointCloudAndPolygon = false;
    bool texturedPolyToggle = true;
    bool semanticMode = false;
    bool semanticToggle = true;
    bool mouseCurrentlyPressed = false;
    std::vector<boost::shared_ptr<SemanticGridEntry> > semanticGridList;
    std::map<std::string, std::vector<boost::shared_ptr<SemanticGridEntry> > > semanticGridMap;

    PointT clickSphereCenter;

    shared_ptr<thread>  viewerThread;
    shared_ptr<thread>  filterThread;
    shared_ptr<pcl::visualization::PCLVisualizer> visualizer;
    shared_ptr<pcl::visualization::PointCloudColorHandlerRGBField<PointT>> colorHandler;
    cv::Vec3d _lastCameraOrientation;
    cv::Vec3d _lastCameraPose;

    int windowWidth;
    int windowHeight;

    // 3 mutex scheme for priority scheduling
    // Low-priority threads: lock L, lock N, lock M, unlock N, { do stuff }, unlock M, unlock L
    // High-priority thread: lock N, lock M, unlock N, { do stuff }, unlock M
    boost::mutex mD;  // mutex for data updates
    boost::mutex mN;  // mutex for next-to-access
    boost::mutex mL;  // mutex for low-priority access
    
    bool updated = true;
    bool filtered = true;
    boost::atomic<bool> visualizerReady;

    bool    shutDownFlag    =false;
    mutex   shutDownMutex;

    condition_variable  keyFrameUpdated;

    boost::condition_variable  visualizerReadyCondition;
    boost::mutex visReadyMutex;

    // data to generate point clouds

    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;


    bool showTraining;
    bool showTest = true;

    double mResolution = 0.1;
    pcl::VoxelGrid<PointT>  voxel;
    
    // corresponds to euler [ x: 1.5707963, y: 0, z: 1.5707963 ]: 
    // 0, -1, 0, 0 
    // 0, 0, -1, 0, 
    // 1, 0, 0, 0, 
    // 0, 0, 0, 1



    // corresponding to [ x: 1.5707963, y: 1.5707963, z: 0 ]:
    // 0, 0, 1, 0 
    // 1, 0, 0, 0 
    // 0, 1, 0, 0 
    // 0, 0, 0, 1
    cv::Mat opticalRotInv2 = (cv::Mat_<double>(4,4) <<
            0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1);


    struct callback_args{
      // structure used to pass arguments to the callback function
      PointCloud::Ptr clicked_points_3d;
      pcl::visualization::PCLVisualizer::Ptr viewerPtr;
    };

static bool isOrigin(coord3_t p) {

    return ((p[0] < 0.001) && (p[0] > -0.001) && (p[1] < 0.001) && (p[1] > -0.001) && (p[2] < 0.001) && (p[2] > -0.001));
}

};

#endif // POINTCLOUDMAPPING_H

