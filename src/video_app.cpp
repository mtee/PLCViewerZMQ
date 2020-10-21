#include <iostream>
#include <fstream>
#include <chrono>

#include "opencv2/core.hpp"

#include "acs_localizer.h"

#include "pointcloudmapping.h"

using namespace std::chrono;


cv::RNG rng(12345);

// function to load pointcloud from visual word assignment (.bin file)
int loadPointCloudFromBIN(std::string filename, PointCloudMapping& cloudMapper, uint32_t nb_clusters){
    uint32_t nb_non_empty_vw, nb_3D_points, nb_descriptors;
    std::vector< cv::Vec3f > points3D;
    std::vector< cv::Vec3b > colors_3D;
    std::vector< unsigned char > all_descriptors;
    std::vector< float > all_descriptors_float;
    std::vector< std::vector< std::pair< uint32_t, uint32_t > > > vw_points_descriptors(nb_clusters);

    std::ifstream ifs(filename, std::ios::in | std::ios::binary);    
    if ( !ifs )
    {
        std::cerr << " ERROR: Cannot read the visual word assignments in " << filename << std::endl;
        return -1;
    }
    uint32_t nb_clusts;
    ifs.read(( char* ) &nb_3D_points, sizeof( uint32_t ) );
    ifs.read(( char* ) &nb_clusts, sizeof( uint32_t ) );
    ifs.read(( char* ) &nb_non_empty_vw, sizeof( uint32_t ) );
    ifs.read(( char* ) &nb_descriptors, sizeof( uint32_t ) );    
    if( nb_clusts != nb_clusters )
        std::cerr << " WARNING: Number of clusters differs! " << nb_clusts << " " << nb_clusters << std::endl;    std::cout << "  Number of non-empty clusters: " << nb_non_empty_vw << " number of points : " << nb_3D_points << " number of descriptors: " << nb_descriptors << std::endl;    // read the 3D points and their visibility polygons

    points3D.resize(nb_3D_points);
    colors_3D.resize(nb_3D_points);
    all_descriptors.resize(128*nb_descriptors);
    float *point_data = new float[3];
    unsigned char *color_data = new unsigned char[3];    
    for( uint32_t i=0; i<nb_3D_points; ++i )
    {
        ifs.read(( char* ) point_data, 3 * sizeof( float ) );
        for( int j=0; j<3; ++j )
        points3D[i][j] = point_data[j];      
        ifs.read(( char* ) color_data, 3 * sizeof( unsigned char ) );
        for( int j=0; j<3; ++j )
        colors_3D[i][j] = color_data[j];
    }
    delete [] point_data;
    delete [] color_data;
    
    cloudMapper.AddPointCloud(points3D, colors_3D, 1);

    return 1;
}



int main(int argc, const char *argv[])
{
    PointCloudMapping cloudMapper(0.05f);
    pca::ACSLocalizer localizer;
    cout << "Localizer initialized" << endl;
    cv::Mat camMatrix_raw = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat camMatrix_undistorted = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat cameraImage;
    cv::Mat colorImage;
    cv::Mat acsTransform = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat acs_opticalRotInv = (cv::Mat_<double>(4,4) <<
            0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1);

    // 360p
    camMatrix_raw.at<double>(0, 0) = 1126.3 / 2;
    camMatrix_raw.at<double>(1, 1) = 1165.46 / 2;
    camMatrix_raw.at<double>(0, 2) = 709.217 / 2;
    camMatrix_raw.at<double>(1, 2) = 356.259 / 2;
    camMatrix_raw.at<double>(2, 2) = 1;

    camMatrix_undistorted.at<double>(0, 0) = 1140 / 2;
    camMatrix_undistorted.at<double>(1, 1) = 1140 / 2;
    camMatrix_undistorted.at<double>(0, 2) = 720 / 2;
    camMatrix_undistorted.at<double>(1, 2) = 360 / 2;
    camMatrix_undistorted.at<double>(2, 2) = 1;
    
    std::string bundle_file("/home/demouser/Documents/RTAB-Map/bundler/office_va9_large/cameras.out"); // path to the bundler file exported by rtabmap
    std::string vw_assignments("/home/demouser/Documents/RTAB-Map/bundler/office_va9_large/bundle.desc_assignments.integer_mean.voctree.clusters.100k.bin"); // path to the bundler file exported by descriptor assignment tool
    std::string cluster_file("/home/demouser/acg_localizer/ACG-localizer/markt_paris_gpu_sift_100k.cluster");    // not model-specific


    //loadPointCloudFromBIN(vw_assignments, cloudMapper, 100000);	// we also can load the feature cloud
    cloudMapper.AddPointCloud("/home/demouser/Documents/RTAB-Map/bundler/office_va9_large/cloud.ply");    // dense cloud exported by rtabmap
   // cloudMapper.AddTexturedPolygonFromOBJ("/home/demouser/Documents/RTAB-Map/bundler/office_va9_large/mesh.obj");	// textured mesh exported in rtabmap
    localizer.init(bundle_file, 100000, cluster_file, vw_assignments, 0, 500, 10);
    std::cout << "starting localizer loop" << std::endl;
    cv::VideoCapture cap("/home/demouser/Documents/RTAB-Map/va_office/test_video/video-175411dfe9d.mp4"); // Change video input path here
    while (true)
    {
       // logger.debug("waiting for request");
        
        cv::Mat frame; // bw input image 
        cap >> frame;
	
        if (frame.empty()) {
            std::cerr << "frame empty" << std::endl;
            break;
        }
        cv::resize(frame, frame, cv::Size(775, 360));
        cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
        cv::undistort(frame, cameraImage, camMatrix_raw, cv::noArray());
        cv::imshow("Query Image", cameraImage);
        cv::waitKey(1);
	// ACS output
        cv::Mat inliers;
        std::vector<float> c2D, c3D;
        cv::Mat mDescriptors_q;
        std::set<size_t> unique_vw;
	//
        cv::Mat tempTransform = localizer.processImage(cameraImage, camMatrix_undistorted, inliers, c2D, c3D, mDescriptors_q, unique_vw);
        uint32_t nb_corr = c2D.size() / 2;
        std::cout << "localizer found " <<  inliers.rows << " inliers " << " and correspondences: " << nb_corr << endl;
        if (tempTransform.empty())
        {
            continue;
        }
        acsTransform = tempTransform;
        cloudMapper.AddOrUpdateFrustum("2", acsTransform.inv(), 0.5, 0, 0, 1, 3, false, acs_opticalRotInv, 0.5, 0.5);
        if (cameraImage.data)
        {
	// Show performance data
            cv::cvtColor(cameraImage, colorImage, cv::COLOR_GRAY2BGR);
            double textSize = 1.0;
            int font = cv::FONT_HERSHEY_PLAIN;
            std::ostringstream str;
            str << "QFs: " << mDescriptors_q.rows;
            cv::putText(colorImage,
                        str.str(),
                        cv::Point(5, 12), // Coordinates
                        font,             // Font
                        textSize,         // Scale. 2.0 = 2x bigger
                        cv::Scalar(0, 255, 0));
            str.str("");
            str.clear();
            str << "VWs: " << unique_vw.size();
            cv::putText(colorImage,
                        str.str(),
                        cv::Point(5, 25),
                        font,
                        textSize,
                        cv::Scalar(0, 255, 0));
            str.str("");
            str.clear();

            str << "Corrs: " << nb_corr;
            cv::putText(colorImage,
                        str.str(),
                        cv::Point(5, 37),
                        font,
                        textSize,
                        cv::Scalar(0, 255, 0));
            str.str("");
            str.clear();
            str << "Inliers: " << inliers.rows;
            cv::putText(colorImage,
                        str.str(),
                        cv::Point(5, 49),
                        font,
                        textSize,
                        cv::Scalar(0, 255, 0));
            str.str("");
            str.clear();
            for (int i = 0; i < c2D.size(); i += 2)
            {
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::circle(colorImage, cv::Point(c2D[i] + (colorImage.cols - 1.0) / 2.0f, -c2D[i + 1] + (colorImage.rows - 1.0) / 2.0f), 4.0, color);
            }
            cv::imshow("Most Recent Localization", colorImage);
            cv::waitKey(1);
        }
    }
    cout << "Press any key to quit.." << endl;
    std::cin.ignore();

    return 0;
}
