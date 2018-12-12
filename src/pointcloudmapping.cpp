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

#include "pointcloudmapping.h"

#include "Eigen/Core"
#include "Eigen/LU"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
//#include <pcl/surface/gp3.h>
//#include <pcl/surface/mls.h>

#include <limits>
#include <vtkCamera.h>
#include <vtkRenderWindow.h>
#include <vtkCubeSource.h>
#include <vtkGlyph3D.h>
#include <vtkGlyph3DMapper.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkImageData.h>
#include <vtkLookupTable.h>
#include <vtkTextureUnitManager.h>
//#include <vtkJPEGReader.h>
//#include <vtkBMPReader.h>
//#include <vtkPNMReader.h>
//#include <vtkPNGReader.h>
//#include <vtkTIFFReader.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPointPicker.h>
#include <vtkCellPicker.h>
#include <vtkTextActor.h>
#include <vtkOBBTree.h>
#include <vtkObjectFactory.h>

#include "vtkImageMatSource.h"

#include <pcl/surface/organized_fast_mesh.h>

static const float frustum_vertices[] = {
    0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f};

static const int frustum_indices[] = {
    1, 2, 3, 4, 1, 0, 2, 0, 3, 0, 4};

void PointCloudMapping::pointPickingEventOccurred(const pcl::visualization::PointPickingEvent &event, void *viewer_void)
{

    float x, y, z;
    if (event.getPointIndex() == -1)
    {
        return;
    }
    event.getPoint(x, y, z);
    clickSphereCenter.x = x;
    clickSphereCenter.y = y;
    clickSphereCenter.z = z;
    clickSphereCenter.r = 255;
    clickSphereCenter.g = 255;
    clickSphereCenter.b = 255;
    if (!this->visualizer->updateSphere(clickSphereCenter, 0.1, 255, 255, 255, "clickSphere"))
        this->visualizer->addSphere(clickSphereCenter, 0.1, 255, 255, 255, "clickSphere");
    std::cout << "[INFO] Point coordinate ( " << x << ", " << y << ", " << z << ")" << std::endl;
}

PointCloudMapping::PointCloudMapping(double resolution_) : showTraining(true), mResolution(resolution_)
{
    visualizerReady = false;
    voxel.setLeafSize(mResolution, mResolution, mResolution);
    globalMap = boost::make_shared<PointCloud>();
    displayCloud = boost::make_shared<PointCloud>();

    //  visualizer = boost::make_shared<pcl::visualization::PCLVisualizer>("viewer");

    viewerThread = make_shared<boost::thread>(bind(&PointCloudMapping::Visualize, this));

    unique_lock<mutex> lock(visReadyMutex);

    while (!visualizerReady)
    {
        visualizerReadyCondition.wait(lock);
    }

    filtered = true;
    updated = true;
}

void PointCloudMapping::Shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    //pcl::io::savePCDFileBinary("test.pcd", globalMap);
    //   cout << "saving to file" << endl;
    //   pcl::io::savePLYFile ("test_pcd.ply", *globalMap, true);
    //   cout << "saved point cloud" << endl;

    visualizer->close();
    viewerThread->join();

    cout << "PCL threads closed" << endl;
}

std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

void PointCloudMapping::AddMeshToPointCloud(cv::Mat &A, cv::Mat transform, cv::Mat image, cv::Mat depth, std::string frustumID)
{
    boost::mutex::scoped_lock lock_L(mL);
    boost::mutex::scoped_lock lock_N(mN);
    boost::mutex::scoped_lock lock_M(mD);
    lock_N.unlock();
    //   std::cout << "adding mesh" << std::endl;

    // remove padding
    //    image = image.rowRange(padding, image.rows - padding).colRange(padding, image.cols - padding);

    // rescale back to standard size
    //  if((image.cols != iWidth) || (image.rows != iHeight)) {
    //      cv::resize(image, image, cv::Size(iWidth, iHeight));
    //  }
    pcl::IndicesPtr indices(new std::vector<int>);

    Eigen::Vector3f viewPoint(transform.at<double>(0, 3), transform.at<double>(1, 3), transform.at<double>(2, 3));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = generatePointCloud(A.at<float>(0, 2), A.at<float>(1, 2), A.at<float>(0, 0), A.at<float>(1, 1), transform, image, depth, true, 1);
    indices->resize(cloud->size());
    for (unsigned int i = 0; i < cloud->size(); ++i)
    {
        indices->at(i) = i;
    }
    //   std::cout << "radius filtering the cloud" << std::endl;

    // cloud = voxelize(cloud, indices, 0.05f);
    //std::cout << "starting radius search. Cloud size: " << cloud->size() <<" . Indices: " << indices->size() <<" . Cloud organized: " << cloud->isOrganized() << " . Cloud is dense: " << cloud->is_dense << std::endl;

    //    indices = radiusFiltering(cloud, indices, 0.5f, 5);
    //   std::cout << "Indices after filtering: " << indices->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output;
    //  std::cout << "extracting indices" << std::endl;
    output = extractIndices(cloud, indices, false, true);
    //   std::cout << "creating Mesh from cloud" << std::endl;

    std::vector<pcl::Vertices> polygons = getOrganizedFastMesh(output, 5 * 3.14 / 180.0, 10, viewPoint);

    if (polygons.size())
    {
        //       std::cout << "polygons are there" << std::endl;
        // remove unused vertices to save memory
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputFiltered(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<pcl::Vertices> outputPolygons;
        std::vector<int> denseToOrganizedIndices = filterNotUsedVerticesFromMesh(*output, polygons, *outputFiltered, outputPolygons);

        pcl::TextureMesh::Ptr textureMesh(new pcl::TextureMesh);
        pcl::toPCLPointCloud2(*outputFiltered, textureMesh->cloud);
        textureMesh->tex_polygons.push_back(outputPolygons);
        int w = cloud->width;
        int h = cloud->height;
        if (w < 1 || h < 1)
            std::cerr << "CLOUD INVALID" << std::endl;
        textureMesh->tex_coordinates.resize(1);
        int nPoints = (int)outputFiltered->size();
        textureMesh->tex_coordinates[0].resize(nPoints);
        for (int i = 0; i < nPoints; ++i)
        {
            //uv
            if (i >= (int)denseToOrganizedIndices.size())
                std::cerr << "CLOUD INVALID" << std::endl;
            int originalVertex = denseToOrganizedIndices[i];
            textureMesh->tex_coordinates[0][i] = Eigen::Vector2f(
                float(originalVertex % w) / float(w),      // u
                float(h - originalVertex / w) / float(h)); // v
        }

        pcl::TexMaterial mesh_material;
        mesh_material.tex_d = 1.0f;
        mesh_material.tex_Ns = 75.0f;
        mesh_material.tex_illum = 1;

        std::stringstream tex_name;
        tex_name << "material_" << frustumID;
        tex_name >> mesh_material.tex_name;

        mesh_material.tex_file = "";

        textureMesh->tex_materials.push_back(mesh_material);
        //      std::cout << "adding textured mesh" << std::endl;
        //TODO:   _cloudViewer->addCloudTextureMesh(cloudName, textureMesh, image, pose);
        addTextureMesh(this->visualizer, textureMesh, image, frustumID, 0);
        //       std::cout << "added textured mesh" << std::endl;
        updated = true;
        filtered = false;
        keyFrameUpdated.notify_one();
    }
    else
    {
        std::cerr << "No triangles created!" << std::endl;
    }
}

void PointCloudMapping::AddTrainingFrameToPointCloud(cv::Mat &A, cv::Mat transform, cv::Mat color, cv::Mat depth, std::string frustumID)
{
    boost::mutex::scoped_lock lock_L(mL);
    boost::mutex::scoped_lock lock_N(mN);
    boost::mutex::scoped_lock lock_M(mD);
    lock_N.unlock();
    //   std::cout << "adding cloud" << std::endl;

    //    std::cout << "rgb size: " << color.size() << std::endl;

    //    ms.debug() << "camera matrix: "  << A.at<float>(0, 2) << " " << A.at<float>(1, 2) << " " << A.at<float>(0, 0) << " " << A.at<float>(1, 1) << "\n";

    //    ms.debug() << "transform inverted\n";

    //PointCloud::Ptr p = generatePointCloud(A.at<float>(0, 2), A.at<float>(1, 2), A.at<float>(0, 0), A.at<float>(1, 1), transform, transform, depth, true);
    PointCloud::Ptr p = generatePointCloud(A.at<float>(0, 2), A.at<float>(1, 2), A.at<float>(0, 0), A.at<float>(1, 1), transform, color, depth.rowRange(0, depth.rows).colRange(0, depth.cols), true, 8, 2.0);
    *globalMap += *p;
    //    ms.debug() <<  "Point cloud generated: " << p->size() << " points." << std::endl;
    //  AddOrUpdateFrustum(frustumID, transform, 0.2, 0, 0, 255);

    //  ms.debug() <<  "Adding frustum:\n" << transform << "\n id: " << frustumID << std::endl;

    //    ms.debug() <<  "Frustum added." << std::endl;
    //  std::cout << "cloud added" << std::endl;

    updated = true;

    if (!this->visualizer->updatePointCloud(globalMap, "cloud"))
    {
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(globalMap);
        this->visualizer->addPointCloud(globalMap, rgb, "cloud");
    }

    filtered = false;

    this->visualizer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8.0,
        "cloud");
    keyFrameUpdated.notify_one();
}

cv::Mat PointCloudMapping::AddTestFrameToPointCloud(cv::Mat &A, cv::Mat transform, cv::Mat color, cv::Mat depth, img_coord_t &cloud, cv::Mat_<cv::Point2i> &sampling, std::string frustumID)
{
    boost::mutex::scoped_lock lock_L(mL);
    boost::mutex::scoped_lock lock_N(mN);
    boost::mutex::scoped_lock lock_M(mD);
    lock_N.unlock();

    // remove padding
    //  color = color.rowRange(padding, color.rows - padding).colRange(padding, color.cols - padding);

    // rescale back to standard size
    //  if((color.cols != iWidth) || (color.rows != iHeight)) {
    //      cv::resize(color, color, cv::Size(iWidth, iHeight));
    //   }
    //    ms.debug() << "transform before inversion: " << tCloned << "\n";
    //  tCloned = tCloned.inv();
    if (!depth.empty())
    {
        //        ms.debug() << "depth provided\n";

        PointCloud::Ptr p = generatePointCloud(A.at<float>(0, 2), A.at<float>(1, 2), A.at<float>(0, 0), A.at<float>(1, 1), transform, color, depth.rowRange(0, depth.rows).colRange(0, depth.cols), false, 8);
        *displayCloud = *p;
    }
    else
    {
        std::cout << "no GT depth provided" << std::endl;
        PointCloud::Ptr tmp(new PointCloud());
        PointT origin;
        origin.x = transform.at<double>(0, 3);
        origin.y = transform.at<double>(1, 3);
        origin.z = transform.at<double>(2, 3);
        //        ms.debug() << "transform: " << tCloned << "\n";
        //        ms.debug() << "origin: " << origin.x << "; " << origin.y << "; " << origin.z << "\n";
        cv::Mat depthMapEstimated(cloud.rows, cloud.cols, CV_64F);
        //        ms.debug() << "reconstructing depth from " << cloud.cols << " by " << cloud.rows << " estimates; type: " << type2str(cloud.type()) << "\n";
        //        ms.debug() << "reconstructing color from " << color.cols << " by " << color.rows << " estimates; type: " << type2str(color.type()) << "\n";
        //        ms.debug() << "sampling matrix size: " << sampling.cols << " by " << sampling.rows << " elements\n";
        for (int i = 0; i < cloud.cols; i++)
        {
            for (int j = 0; j < cloud.rows; j++)
            {
                PointT p;
                p.x = (float)(cloud.at<coord3_t>(j, i)[0]); // 1000;
                p.y = (float)(cloud.at<coord3_t>(j, i)[1]); // 1000;
                p.z = (float)(cloud.at<coord3_t>(j, i)[2]); // 1000;

                float dist = std::sqrt(pcl::geometry::squaredDistance(origin, p));
                depthMapEstimated.at<double>(j, i) = dist;

                p.b = color.at<cv::Vec<uchar, 3>>(sampling.at<cv::Point2i>(j, i).y, sampling.at<cv::Point2i>(j, i).x)[0];
                p.g = color.at<cv::Vec<uchar, 3>>(sampling.at<cv::Point2i>(j, i).y, sampling.at<cv::Point2i>(j, i).x)[1];
                p.r = color.at<cv::Vec<uchar, 3>>(sampling.at<cv::Point2i>(j, i).y, sampling.at<cv::Point2i>(j, i).x)[2];
                p.a = 255;
                tmp->points.push_back(p);
            }
        }
        //        ms.debug() << "estimated depth point cloud has " << tmp->points.size() << " points\n";

        double minVal;
        double maxVal;
        cv::minMaxLoc(depthMapEstimated, &minVal, &maxVal);
        //        ms.debug() << "depthmap complete\n min: " << minVal << " max: " << maxVal << std::endl;

        // rescale color space so it fills the range [0; 255]
        cv::Mat adjMap;
        cv::convertScaleAbs(depthMapEstimated, adjMap, 255 / 8);
        cv::resize(adjMap, adjMap, cv::Size(), 8, 8, cv::INTER_NEAREST);
        cv::imshow("depthmap estimated", adjMap);
        cv::waitKey(5);
        *displayCloud = *tmp;
    }
    updated = true;
    filtered = false;
    //    ms.debug() << "transform by which we transform the cloud: " << tCloned << "\n";
    keyFrameUpdated.notify_one();
    return transform;
}


void PointCloudMapping::Add2D3DCorrespondencesToPointCloud(std::vector< float > c3D) {
    PointCloud::Ptr tmp(new PointCloud());
    cv::RNG rng(12345);
    for (int i = 0; i < c3D.size(); i+=3) {
        PointT p;
        p.x = c3D[i];
        p.y = c3D[i+1];
        p.z = c3D[i+2];
        p.r = rng.uniform(0, 255);
        p.g = rng.uniform(0, 255);
        p.b = rng.uniform(0, 255);
        p.a = 255;
        tmp->points.push_back(p);
    }
    *displayCloud = *tmp;
}

void PointCloudMapping::AddPointCloud(std::string filename) {
    boost::mutex::scoped_lock lock_L(mL);
    boost::mutex::scoped_lock lock_N(mN);
    boost::mutex::scoped_lock lock_M(mD);
    lock_N.unlock();

    PointCloud::Ptr tmp(new PointCloud());
    if (pcl::io::loadPLYFile<PointT> ("cloud.ply", *tmp) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cout << "Loaded "
                << tmp->width * tmp->height
                << " data points from cloud.ply"
                << std::endl;
    tmp->is_dense = false;

    *globalMap += *tmp;

    updated = true;
    filtered = true;

    // cv::Mat cameraMatrix = AddOrUpdateFrustum(frustumId, tCloned, 0.5, 255, 0, 0);
    if (!this->visualizer->updatePointCloud(globalMap, "cloud"))
    {
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(globalMap);
        this->visualizer->addPointCloud(globalMap, rgb, "cloud");
    }

    // performance test
    // int CUBE_NUM = 40;
    // float cube_size = 0.05;
    // float cube_distance = 0.1;
    // for (int i = 0; i < CUBE_NUM; ++i)
    // {
    //     for (int j = 0; j < CUBE_NUM; ++j)
    //     {
    //         cv::waitKey(5);
    //         visualizer->addCube(i * cube_distance, i * cube_distance + cube_size, j * cube_distance, j * cube_distance + cube_size, 0, cube_size, 1.0, 1.0, 1.0, std::to_string(i * CUBE_NUM + j), 0);
    //     }
    // }

    std::cout << "added" << std::endl;

    lock_M.unlock();
    lock_L.unlock();
    keyFrameUpdated.notify_one();
}

void PointCloudMapping::AddPointCloud(std::vector<cv::Vec3f> points, std::vector<cv::Vec3b> colorList, float scale)
{
    boost::unique_lock<mutex> updateLock(mD);
    std::cout << "adding point cloud with points: " << points.size() << std::endl;
    //    ms << "Adding test frame to the cloud\n";

    // remove padding
    //      color = color.rowRange(padding, color.rows - padding).colRange(padding, color.cols - padding);

    // rescale back to standard size
    //      if((color.cols != iWidth) || (color.rows != iHeight)) {
    //          cv::resize(color, color, cv::Size(iWidth, iHeight));
    //       }

    PointCloud::Ptr tmp(new PointCloud());
    float range = 1000;
    for (int i = 0; i < points.size(); i++)
    {
        PointT p;
        p.x = (double)points[i][0] / scale; 
        p.y = (double)points[i][1] / scale; 
        p.z = (double)points[i][2] / scale;

        p.b = (unsigned char) colorList[i][2];
        p.g = (unsigned char) colorList[i][1];
        p.r = (unsigned char) colorList[i][0];

        p.a = 255;
        if (p.x < range && p.y < range && p.x > -range && p.y > -range)
            tmp->points.push_back(p);
    }
    tmp->is_dense = false;

    *globalMap += *tmp;

    updated = true;
    filtered = true;

    // cv::Mat cameraMatrix = AddOrUpdateFrustum(frustumId, tCloned, 0.5, 255, 0, 0);
    if (!this->visualizer->updatePointCloud(globalMap, "cloud"))
    {
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(globalMap);
        this->visualizer->addPointCloud(globalMap, rgb, "cloud");
    }
std::cout << "added" << std::endl;
    keyFrameUpdated.notify_one();
}

void PointCloudMapping::AddPointCloud(cv::Mat transform, cv::Mat color, img_coord_t &cloud, std::string frustumId)
{
    boost::unique_lock<mutex> updateLock(mD);
    
    //    ms << "Adding test frame to the cloud\n";

    // remove padding
    //      color = color.rowRange(padding, color.rows - padding).colRange(padding, color.cols - padding);

    // rescale back to standard size
    //      if((color.cols != iWidth) || (color.rows != iHeight)) {
    //          cv::resize(color, color, cv::Size(iWidth, iHeight));
    //       }
    cv::Mat_<double> tCloned = transform.clone();
    //    ms.debug() << "transform before inversion: " << tCloned << "\n";
    //  tCloned = tCloned.inv();
    //    ms.debug() << "no GT depth provided\n";
    PointCloud::Ptr tmp(new PointCloud());
    PointT origin;
    origin.x = tCloned(0, 3);
    origin.y = tCloned(1, 3);
    origin.z = tCloned(2, 3);
    //    ms.debug() << "transform: " << tCloned << "\n";
    //    ms.debug() << "origin: " << origin.x << "; " << origin.y << "; " << origin.z << "\n";
    cv::Mat depthMapEstimated = cv::Mat::zeros(cloud.rows, cloud.cols, CV_64F);
    //    ms.debug() << "reconstructing depth from " << cloud.cols << " by " << cloud.rows << " estimates; type: " << type2str(cloud.type()) << "\n";
    //    ms.debug() << "reconstructing color from " << color.cols << " by " << color.rows << " estimates; type: " << type2str(color.type()) << "\n";
    for (int i = 0; i < cloud.cols; i++)
    {
        for (int j = 0; j < cloud.rows; j++)
        {
            if (!isOrigin(cloud.at<coord3_t>(j, i)))
            {
                PointT p;
                p.x = (double)(cloud.at<coord3_t>(j, i)[0]); // 1000;
                p.y = (double)(cloud.at<coord3_t>(j, i)[1]); // 1000;
                p.z = (double)(cloud.at<coord3_t>(j, i)[2]); // 1000;

                double dist = std::sqrt(pcl::geometry::squaredDistance(origin, p));
                depthMapEstimated.at<double>(j, i) = dist;

                p.b = color.at<bgr_t>(j, i)[0];
                p.g = color.at<bgr_t>(j, i)[1];
                p.r = color.at<bgr_t>(j, i)[2];
                p.a = 255;
                tmp->points.push_back(p);
            }
        }
    }
    tmp->is_dense = false;
    //    ms.debug() << "estimated depth point cloud has " << tmp->points.size() << " points\n";

    //    double minVal;
    //    double maxVal;
    //    cv::minMaxLoc(depthMapEstimated, &minVal, &maxVal);

    // rescale color space so it fills the range [0; 255]
    //    cv::Mat adjMap;
    //    cv::convertScaleAbs(depthMapEstimated, adjMap, 255 / 8);
    //  cv::resize(adjMap, adjMap, cv::Size(), 8, 8, cv::INTER_NEAREST);
    //    cv::imshow("depthmap reconstructed", adjMap);
    //    cv::waitKey(1);
    //    cv::imshow("color", color);
    //    cv::waitKey(1);

    *globalMap += *tmp;

    updated = true;
    filtered = false;

    // cv::Mat cameraMatrix = AddOrUpdateFrustum(frustumId, tCloned, 0.5, 255, 0, 0);
    if (!this->visualizer->updatePointCloud(globalMap, "cloud"))
    {
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(globalMap);
        this->visualizer->addPointCloud(globalMap, rgb, "cloud");
    }

    keyFrameUpdated.notify_one();
}

unsigned int text_id = 0;
void PointCloudMapping::keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                                              void *viewer_void)
{
    std::cout << " keyboard callback called" << std::endl;
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
    /*  if (event.getKeySym () == "w" && event.keyDown ()) {
      pcl::visualization::Camera cam;
      viewer->getCameraParameters(cam);
      double delta = 0.1;
      cam.focal[2] += delta;
      viewer->setCameraParameters(cam, 0);

      cout << "Cam: " << endl
                   << " - pos: (" << cam.pos[0] << ", "    << cam.pos[1] << ", "    << cam.pos[2] << ")" << endl
                   << " - view: ("    << cam.view[0] << ", "   << cam.view[1] << ", "   << cam.view[2] << ")"    << endl
                   << " - focal: ("   << cam.focal[0] << ", "  << cam.focal[1] << ", "  << cam.focal[2] << ")"   << endl;

    //  viewer->setCameraPosition(camera.pos[0] + 0.1, camera.pos[1], camera.pos[2], camera.view[0], camera.view[1], camera.view[2], 0);

  }

  if (event.getKeySym () == "a" && event.keyDown ()) {
      pcl::visualization::Camera cam;
      viewer->getCameraParameters(cam);
      double delta = 0.1;
      cam.focal[0] -= delta;
      viewer->setCameraParameters(cam, 0);
  }
  if (event.getKeySym () == "d" && event.keyDown ()) {
          pcl::visualization::Camera cam;
          viewer->getCameraParameters(cam);
          double delta = 0.1;
          cam.focal[0] += delta;
          viewer->setCameraParameters(cam, 0);
    } */
    if (event.getKeySym() == "e" && event.keyDown())
    {
        showTest = !showTest;
        updated = true;
        filtered = true;
        keyFrameUpdated.notify_one();
        std::cout << "keypress detected" << std::endl;
    }
    if (event.getKeySym() == "t" && event.keyDown())
    {
        showTraining = !showTraining;
        updated = true;
        filtered = true;
        keyFrameUpdated.notify_one();
        std::cout << "keypress detected" << std::endl;
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudMapping::generatePointCloud(float cx, float cy, float fx, float fy, cv::Mat transform, cv::Mat color, cv::Mat depth, bool gtData, int decimateDepth, float depthMax)
{
    PointCloud::Ptr tmp(new PointCloud());
    tmp->width = depth.cols;
    tmp->height = depth.rows;
    tmp->resize(tmp->height * tmp->width);
    // point cloud is null ptr
    const float bad_point = std::numeric_limits<float>::quiet_NaN();

    for (int m = 0; m < depth.rows; m += decimateDepth)
    {
        for (int n = 0; n < depth.cols; n += decimateDepth)
        {
            double dint = depth.at<double>(m, n);
            double d = (double)dint / 1000;
            PointT p;
            if (d < 0.01 || d > depthMax)
            {
                p.x = bad_point;
                p.y = bad_point;
                p.z = bad_point;
            }
            else
            {
                p.z = d; // flipping z
                p.x = ((n)-cx) * p.z / (fx);
                p.y = ((m)-cy) * p.z / (fy);

                p.b = color.ptr<uchar>(m)[n * 3];
                p.g = color.ptr<uchar>(m)[n * 3 + 1];
                p.r = color.ptr<uchar>(m)[n * 3 + 2];

                tmp->points.at(m * tmp->width + n) = p;
            }
        }
    }
    Eigen::Matrix<double, 4, 4> T;
    cv::cv2eigen(transform, T);
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud(*tmp, *cloud, T);

    //  pcl::transformPointCloud(*tmp, *cloud, mBasis);
    cloud->is_dense = false;
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudMapping::voxelize(
    const typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    const pcl::IndicesPtr &indices,
    float voxelSize)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
    if ((cloud->is_dense && cloud->size()) || (!cloud->is_dense && indices->size()))
    {
        pcl::VoxelGrid<PointT> filter;
        filter.setLeafSize(voxelSize, voxelSize, voxelSize);
        filter.setInputCloud(cloud);
        if (indices->size())
        {
            filter.setIndices(indices);
        }
        filter.filter(*output);
    }
    else if (cloud->size() && !cloud->is_dense && indices->size() == 0)
    {
        std::cerr << "Cannot voxelize a not dense (organized) cloud with empty indices! Returning empty cloud!" << std::endl;
    }
    return output;
}

void PointCloudMapping::OptimizePointCloud()
{
    while (!shutDownFlag)
    {

        unique_lock<mutex> lck_keyframeUpdated(mD);
        keyFrameUpdated.wait(lck_keyframeUpdated);
        if (globalMap)
            if (globalMap->points.size() > 0)
            {
                if (updated && !filtered)
                {
                    PointCloud::Ptr tmp(new PointCloud());
                    voxel.setInputCloud(globalMap);
                    voxel.filter(*tmp);
                    globalMap->swap(*tmp);

                    //		updateLock.unlock();
                    lck_keyframeUpdated.unlock();
                    filtered = true;
                    //       updated = false;
                }
            }
            else
            {
                filtered = true;
            }

        //    std::cout << "filtering done" << std::endl;
    }
}

void PointCloudMapping::RemoveFrustum(const std::string &id)
{
    if (id.empty())
    {
        std::cout << "id should not be empty!" << std::endl;
        ;
        return;
    }
    frustumIterator = _frustums.find(id);
    if (frustumIterator != _frustums.end())
    {
        _frustums.erase(frustumIterator);
        visualizer->removeShape(id);
    }
    else
        std::cout << "FRUSTUM NOT FOUND" << std::endl;
}


//lock N, lock M, unlock N, { do stuff }, unlock M
void PointCloudMapping::AddOrUpdateFrustum(
    const std::string &id,
    const cv::Mat &transform,
    float scale, double r, double g, double b, float lineWidth)
{

    boost::mutex::scoped_lock lock_N(mN);
    boost::mutex::scoped_lock lock_M(mD);
    lock_N.unlock();

    if (id.empty())
    {
        std::cout << "FRUSTUM ID SHOULD NOT BE EMPTY" << std::endl;
        return;
    }


    bool newFrustum = _frustums.find(id) == _frustums.end();
    if (!newFrustum)
    {
        this->RemoveFrustum(id);
        newFrustum = true;
    }

    if (!transform.empty())
    {
        if (newFrustum)
        {
            _frustums.insert(std::make_pair(id, cv::Mat::eye(4, 4, CV_32F)));

            int frustumSize = sizeof(frustum_vertices) / sizeof(float);
            if (!(frustumSize > 0 && frustumSize % 3 == 0))
                std::cout << "FRUSTUM SIZE IS WRONG" << std::endl;
            frustumSize /= 3;
            pcl::PointCloud<pcl::PointXYZ> frustumPoints;
            frustumPoints.resize(frustumSize);
            float scaleX = 0.5f * scale;
            float scaleY = 0.4f * scale; //4x3 arbitrary ratio
            float scaleZ = 0.3f * scale;
            // Transform

            Eigen::Transform<float, 3, Eigen::Affine> t;
            

            //cv::Mat transformResult = transform.inv() * opticalRotInv;
            cv::Mat transformResult = transform * opticalRotInv;
            for (int i = 0; i < 4; i++)
            {
                t(i, 0) = (float)transformResult.at<double>(i, 0);
                t(i, 1) = (float)transformResult.at<double>(i, 1);
                t(i, 2) = (float)transformResult.at<double>(i, 2);
                t(i, 3) = (float)transformResult.at<double>(i, 3);
            }

            for (int i = 0; i < frustumSize; ++i)
            {
                frustumPoints[i].x = frustum_vertices[i * 3] * scaleX;
                frustumPoints[i].y = frustum_vertices[i * 3 + 1] * scaleY;
                frustumPoints[i].z = frustum_vertices[i * 3 + 2] * scaleZ;
                frustumPoints[i] = pcl::transformPoint(frustumPoints[i], t);
            }

            //    std::vector<pcl::visualization::Camera> cameras;
            //    this->visualizer->getCameras(cameras);

            //            this->visualizer->setCameraPosition(
            //                t(0, 3), t(1, 3), t(2, 3),
            //                cameras.front().focal[0], cameras.front().focal[1], cameras.front().focal[2],
            //                cameras.front().view[0], cameras.front().view[1], cameras.front().view[2]);

            pcl::PolygonMesh mesh;
            pcl::Vertices vertices;
            vertices.vertices.resize(sizeof(frustum_indices) / sizeof(int));
            for (unsigned int i = 0; i < vertices.vertices.size(); ++i)
            {
                vertices.vertices[i] = frustum_indices[i];
            }
            pcl::toPCLPointCloud2(frustumPoints, mesh.cloud);
            mesh.polygons.push_back(vertices);
            //            ms.trace() << "adding frustum mesh: " << mesh.cloud.data.size() << " vertices size: " << vertices.vertices.size() << std::endl;
            visualizer->addPolylineFromPolygonMesh(mesh, id);
            //            ms.trace() << "polyline added" << std::endl;
            visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
            //visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 20, id);
            visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, lineWidth, id);

            //            ms.trace() << "frustum mesh added" << std::endl;
        }
        lock_M.unlock();
    }
    else
    {
        std::cout << "no frustum transform supplied, removing ID" << std::endl;
        RemoveFrustum(id);
    }
    return;
}

cv::Vec3f PointCloudMapping::GetCloudCentroid()
{
    PointT minPt;
    PointT maxPt;
    pcl::getMinMax3D<PointT>(*globalMap, minPt, maxPt);
    std::cout << "min point: " << minPt << std::endl;
    std::cout << "max point: " << maxPt << std::endl;
    return cv::Vec3f((minPt.x + maxPt.x) / 2, (minPt.y + maxPt.y) / 2, (minPt.z + maxPt.z) / 2);
}

void PointCloudMapping::Visualize()
{
    // prepare visualizer named "viewer"
    this->visualizer = boost::make_shared<pcl::visualization::PCLVisualizer>("viewer");
    unique_lock<mutex> lock(visReadyMutex);
    visualizerReady = true;
    lock.unlock();
    visualizerReadyCondition.notify_one();

    this->visualizer->setBackgroundColor(0, 0, 0);
    this->visualizer->registerKeyboardCallback(&PointCloudMapping::keyboardEventOccurred, *this);
    this->visualizer->registerPointPickingCallback(&PointCloudMapping::pointPickingEventOccurred, *this);
    this->visualizer->addCoordinateSystem(1.0);
    this->visualizer->setCameraClipDistances(0.001, 1000);
    // x=red axis, y=green axis, z=blue axis z direction is point into the screen.
    //  this->visualizer->addCoordinateSystem(1.0);
    pcl::ModelCoefficients sphere_coeff;
    sphere_coeff.values.resize(4); // We need 4 values
    sphere_coeff.values[0] = 0;
    sphere_coeff.values[1] = 0;
    sphere_coeff.values[2] = 0;
    sphere_coeff.values[3] = 0.1f;
    //  this->visualizer->addSphere(sphere_coeff, "clickSphere");

    displayCloud->width = (int)displayCloud->points.size();
    displayCloud->height = 1;

    globalMap->width = (int)globalMap->points.size();
    globalMap->height = 1;

    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(globalMap);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbDisplay(displayCloud);
    filterThread = make_shared<thread>(bind(&PointCloudMapping::OptimizePointCloud, this));

    while (!shutDownFlag)
    {
        //    std::cout << "vis start" << std::endl;
        boost::mutex::scoped_lock lock_L(mL);
        boost::mutex::scoped_lock lock_N(mN);
        boost::mutex::scoped_lock lock_M(mD);
        lock_N.unlock();
        this->visualizer->spinOnce(5);

        // Get lock on the boolean update and check if cloud was updated

        if (filtered && updated)
        {
            // boost::mutex::scoped_lock updateLock(mD);

            this->visualizer->updateSphere(this->clickSphereCenter, 0.1, 255, 255, 255, "clickSphere");
            //            if (showTraining) {
            //                if (!this->visualizer->updatePointCloud(globalMap, "cloud")) {
            //                    this->visualizer->addPointCloud(globalMap, rgb, "cloud");
            //                    this->visualizer->setPointCloudRenderingProperties(
            //                                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6.0,
            //                                "cloud");
            //                }
            //            } else {
            //                this->visualizer->removePointCloud("cloud");
            //            }
            if (showTest)
            {
                if (!visualizer->updatePointCloud(displayCloud, "curFrame"))
                {
                    visualizer->addPointCloud(displayCloud, rgbDisplay, "curFrame");

                    visualizer->setPointCloudRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0,
                        "curFrame");
                }
            }
            updated = false;
        }
        lock_M.unlock();
        lock_L.unlock();
        //     std::cout << "vis end" << std::endl;
    }
    filterThread->join();
    //   std::cout << "vis end" << std::endl;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudMapping::extractIndices(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    const pcl::IndicesPtr &indices,
    bool negative,
    bool keepOrganized)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.setKeepOrganized(keepOrganized);
    extract.filter(*output);
    return output;
}

pcl::IndicesPtr PointCloudMapping::radiusFiltering(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    const pcl::IndicesPtr &indices,
    float radiusSearch,
    int minNeighborsInRadius)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>(false));

    if (indices->size())
    {
        pcl::IndicesPtr output(new std::vector<int>(indices->size()));
        int oi = 0; // output iterator
        tree->setInputCloud(cloud, indices);
        //      std::cout << "starting radius search. Cloud size: " << cloud->size() <<" . Indices: " << indices->size() << ". Radius: " << radiusSearch << " . Cloud organized: " << cloud->isOrganized() << " . Cloud is dense: " << cloud->is_dense << std::endl;
        for (unsigned int i = 0; i < indices->size(); ++i)
        {
            std::vector<int> kIndices;
            std::vector<float> kDistances;

            int k = tree->radiusSearch(cloud->at(indices->at(i)), radiusSearch, kIndices, kDistances);
            if (k > minNeighborsInRadius)
            {
                output->at(oi++) = indices->at(i);
            }
        }
        output->resize(oi);
        return output;
    }
    else
    {
        pcl::IndicesPtr output(new std::vector<int>(cloud->size()));
        int oi = 0; // output iterator
        tree->setInputCloud(cloud);
        for (unsigned int i = 0; i < cloud->size(); ++i)
        {
            std::vector<int> kIndices;
            std::vector<float> kDistances;
            int k = tree->radiusSearch(cloud->at(i), radiusSearch, kIndices, kDistances);
            if (k > minNeighborsInRadius)
            {
                output->at(oi++) = i;
            }
        }
        output->resize(oi);
        return output;
    }
}

std::vector<pcl::Vertices> PointCloudMapping::getOrganizedFastMesh(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    double angleTolerance,
    int trianglePixelSize,
    const Eigen::Vector3f &viewpoint)
{
    //   std::cout << "creating mesh from cloud with " << cloud->size() << " points, from the viewpoint: " << viewpoint  << std::endl;
    //   std::cout << "cloud is organized: " << cloud->isOrganized() << std::endl;
    pcl::OrganizedFastMesh<pcl::PointXYZRGB> ofm;
    ofm.setTrianglePixelSize(trianglePixelSize);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<pcl::PointXYZRGB>::QUAD_MESH);
    ofm.setInputCloud(cloud);
    //TODO: check whether PCL 1.8 can be built to use these:
    //   ofm.setAngleTolerance(angleTolerance);
    //   ofm.setViewpoint(viewpoint);

    std::vector<pcl::Vertices> vertices;
    ofm.reconstruct(vertices);
    //   std::cout << "created mesh has vertices: " << vertices.size() << std::endl;

    //flip all polygons (right handed)
    std::vector<pcl::Vertices> output(vertices.size());
    for (unsigned int i = 0; i < vertices.size(); ++i)
    {
        output[i].vertices.resize(4);
        output[i].vertices[0] = vertices[i].vertices[0];
        output[i].vertices[3] = vertices[i].vertices[1];
        output[i].vertices[2] = vertices[i].vertices[2];
        output[i].vertices[1] = vertices[i].vertices[3];
    }

    return output;
}
std::vector<int> PointCloudMapping::filterNotUsedVerticesFromMesh(
    const pcl::PointCloud<pcl::PointXYZRGB> &cloud,
    const std::vector<pcl::Vertices> &polygons,
    pcl::PointCloud<pcl::PointXYZRGB> &outputCloud,
    std::vector<pcl::Vertices> &outputPolygons)
{
    std::map<int, int> addedVertices; //<oldIndex, newIndex>
    std::vector<int> output;          //<oldIndex>
    output.resize(cloud.size());
    outputCloud.resize(cloud.size());
    outputCloud.is_dense = true;
    outputPolygons.resize(polygons.size());
    int oi = 0;
    for (unsigned int i = 0; i < polygons.size(); ++i)
    {
        pcl::Vertices &v = outputPolygons[i];
        v.vertices.resize(polygons[i].vertices.size());
        for (unsigned int j = 0; j < polygons[i].vertices.size(); ++j)
        {
            std::map<int, int>::iterator iter = addedVertices.find(polygons[i].vertices[j]);
            if (iter == addedVertices.end())
            {
                outputCloud[oi] = cloud.at(polygons[i].vertices[j]);
                addedVertices.insert(std::make_pair(polygons[i].vertices[j], oi));
                output[oi] = polygons[i].vertices[j];
                v.vertices[j] = oi++;
            }
            else
            {
                v.vertices[j] = iter->second;
            }
        }
    }
    outputCloud.resize(oi);
    output.resize(oi);

    return output;
}

bool PointCloudMapping::addTextureMesh(
    shared_ptr<pcl::visualization::PCLVisualizer> _visualizer,
    pcl::TextureMesh::Ptr mesh,
    const cv::Mat &image,
    const std::string &id,
    int viewport)
{
    // Copied from PCL 1.8, modified to ignore vertex color and accept only one material (loaded from memory instead of file)

    pcl::visualization::CloudActorMap::iterator am_it = _visualizer->getCloudActorMap()->find(id);
    if (am_it != _visualizer->getCloudActorMap()->end())
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] A shape with id <%s> already exists!"
                  " Please choose a different id and retry.\n",
                  id.c_str());
        return (false);
    }
    // no texture materials --> exit
    if (mesh->tex_materials.size() == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] No textures found!\n");
        return (false);
    }
    else if (mesh->tex_materials.size() > 1)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] only one material per mesh is supported!\n");
        return (false);
    }
    // polygons are mapped to texture materials

    if (mesh->tex_materials.size() != mesh->tex_polygons.size())
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] Materials number %lu differs from polygons number %lu!\n",
                  mesh->tex_materials.size(), mesh->tex_polygons.size());
        return (false);
    }
    // each texture material should have its coordinates set
    if (mesh->tex_materials.size() != mesh->tex_coordinates.size())
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] Coordinates number %lu differs from materials number %lu!\n",
                  mesh->tex_coordinates.size(), mesh->tex_materials.size());
        return (false);
    }
    // total number of vertices
    std::size_t nb_vertices = 0;
    for (std::size_t i = 0; i < mesh->tex_polygons.size(); ++i)
        nb_vertices += mesh->tex_polygons[i].size();
    // no vertices --> exit
    if (nb_vertices == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] No vertices found!\n");
        return (false);
    }
    // total number of coordinates
    std::size_t nb_coordinates = 0;
    for (std::size_t i = 0; i < mesh->tex_coordinates.size(); ++i)
        nb_coordinates += mesh->tex_coordinates[i].size();
    // no texture coordinates --> exit
    if (nb_coordinates == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] No textures coordinates found!\n");
        return (false);
    }

    // Create points from mesh.cloud
    vtkSmartPointer<vtkPoints> poly_points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    bool has_color = false;
    vtkSmartPointer<vtkMatrix4x4> transformation = vtkSmartPointer<vtkMatrix4x4>::New();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromPCLPointCloud2(mesh->cloud, *cloud);
    // no points --> exit
    if (cloud->points.size() == 0)
    {
        PCL_ERROR("[PCLVisualizer::addTextureMesh] Cloud is empty!\n");
        return (false);
    }
    pcl::visualization::PCLVisualizer::convertToVtkMatrix(cloud->sensor_origin_, cloud->sensor_orientation_, transformation);
    poly_points->SetNumberOfPoints(cloud->points.size());
    for (std::size_t i = 0; i < cloud->points.size(); ++i)
    {
        const pcl::PointXYZ &p = cloud->points[i];
        poly_points->InsertPoint(i, p.x, p.y, p.z);
    }

    //create polys from polyMesh.tex_polygons
    vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
    for (std::size_t i = 0; i < mesh->tex_polygons.size(); i++)
    {
        for (std::size_t j = 0; j < mesh->tex_polygons[i].size(); j++)
        {
            std::size_t n_points = mesh->tex_polygons[i][j].vertices.size();
            polys->InsertNextCell(int(n_points));
            for (std::size_t k = 0; k < n_points; k++)
                polys->InsertCellPoint(mesh->tex_polygons[i][j].vertices[k]);
        }
    }

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPolys(polys);
    polydata->SetPoints(poly_points);
    if (has_color)
        polydata->GetPointData()->SetScalars(colors);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION < 6
    mapper->SetInput(polydata);
#else
    mapper->SetInputData(polydata);
#endif

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    vtkTextureUnitManager *tex_manager = vtkOpenGLRenderWindow::SafeDownCast(_visualizer->getRenderWindow())->GetTextureUnitManager();
    if (!tex_manager)
        return (false);

    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    // fill vtkTexture from pcl::TexMaterial structure
    vtkSmartPointer<vtcv::vtkImageMatSource> cvImageToVtk = vtkSmartPointer<vtcv::vtkImageMatSource>::New();
    cvImageToVtk->SetImage(image);
    cvImageToVtk->Update();
    texture->SetInputConnection(cvImageToVtk->GetOutputPort());

    // set texture coordinates
    vtkSmartPointer<vtkFloatArray> coordinates = vtkSmartPointer<vtkFloatArray>::New();
    coordinates->SetNumberOfComponents(2);
    coordinates->SetNumberOfTuples(mesh->tex_coordinates[0].size());
    for (std::size_t tc = 0; tc < mesh->tex_coordinates[0].size(); ++tc)
    {
        const Eigen::Vector2f &uv = mesh->tex_coordinates[0][tc];
        coordinates->SetTuple2(tc, (double)uv[0], (double)uv[1]);
    }
    coordinates->SetName("TCoords");
    polydata->GetPointData()->SetTCoords(coordinates);
    // apply texture
    actor->SetTexture(texture);

    // set mapper
    actor->SetMapper(mapper);

    //_visualizer->addActorToRenderer (actor, viewport);
    // Add it to all renderers
    _visualizer->getRendererCollection()->InitTraversal();
    vtkRenderer *renderer = NULL;
    int i = 0;
    while ((renderer = _visualizer->getRendererCollection()->GetNextItem()) != NULL)
    {
        // Should we add the actor to all renderers?
        if (viewport == 0)
        {
            renderer->AddActor(actor);
        }
        else if (viewport == i) // add the actor only to the specified viewport
        {
            renderer->AddActor(actor);
        }
        ++i;
    }

    // Save the pointer/ID pair to the global actor map
    (*_visualizer->getCloudActorMap())[id].actor = actor;

    // Save the viewpoint transformation matrix to the global actor map
    (*_visualizer->getCloudActorMap())[id].viewpoint_transformation_ = transformation;

    _visualizer->getCloudActorMap()->find(id)->second.actor->GetProperty()->SetLighting(false);
    _visualizer->getCloudActorMap()->find(id)->second.actor->GetProperty()->SetInterpolation(VTK_PHONG);
    _visualizer->getCloudActorMap()->find(id)->second.actor->GetProperty()->SetEdgeVisibility(false);
    _visualizer->getCloudActorMap()->find(id)->second.actor->GetProperty()->SetBackfaceCulling(true);
    _visualizer->getCloudActorMap()->find(id)->second.actor->GetProperty()->SetFrontfaceCulling(false);
    return true;
}

