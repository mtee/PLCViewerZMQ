#include <iostream>
#include <fstream>
#include <chrono>

#include "opencv2/core.hpp"

#include "acs_localizer.h"

#include "pointcloudmapping.h"

using namespace std::chrono;




 cv::RNG rng(12345);
// substract 1548688800000 to get seconds


inline double toDouble(std::string s)
{
    std::replace(s.begin(), s.end(), ',', '.');
    return std::atof(s.c_str());
}

template <typename T>
void pop_front(std::vector<T> &vec)
{
    assert(!vec.empty());
    vec.erase(vec.begin());
}

// returns the index of the closest element of a vector to the given value
int closest(std::vector<uint64> const &vec, uint64 value)
{
    //  auto const itL = std::lower_bound(vec.begin(), vec.end(), value);
    auto const itU = std::upper_bound(vec.begin(), vec.end(), value);
    // itU points to the first element bigger than value.
    // we have to find out whether itU or its predecessor is closer to value
    int upperInd = itU - vec.begin();
    if (upperInd > 0)
    {
        if (vec[upperInd] - value > vec[upperInd - 1] - value)
            return upperInd - 1;
        else
            return upperInd;
    }
    return upperInd;
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim))
        elems.push_back(item);

    return elems;
}

std::vector<std::string> split(const std::string &s)
{
    std::istringstream iss(s);
    std::vector<std::string> elems;

    std::copy(
        std::istream_iterator<std::string>(iss),
        std::istream_iterator<std::string>(),
        std::back_inserter<std::vector<std::string>>(elems));

    return elems;
}


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
    cout << "Main started" << endl;
    PointCloudMapping cloudMapper(0.05f);
    pca::ACSLocalizer localizer;
    cout << "Localizer initialized" << endl;
    bool localizerWorking;
    bool poseValid = false;
    cv::Mat camMatrix_raw = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat camMatrix_undistorted = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat cameraImage;
    cv::Mat colorImage;
    cv::Mat arcoreTransform = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat arcoreDiff = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat arcoreSnapshot = cv::Mat::eye(4, 4, CV_64F); // snapshot of the arcore pose valid at the moment of the most recent localization
    cv::Mat acsTransform = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat acs_opticalRotInv = (cv::Mat_<double>(4,4) <<
            0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1);
    // these numbers are suitable for 640x480 images that come from the arcore-fusion-unity-app
    // camMatrix_raw.at<double>(0, 0) = 489.776;
    // camMatrix_raw.at<double>(1, 1) = 493.376;
    // camMatrix_raw.at<double>(0, 2) = 316.348;
    // camMatrix_raw.at<double>(1, 2) = 238.132;
    // camMatrix_raw.at<double>(2, 2) = 1;

    // camMatrix_undistorted.at<double>(0, 0) = 490;
    // camMatrix_undistorted.at<double>(1, 1) = 490;
    // camMatrix_undistorted.at<double>(0, 2) = 320;
    // camMatrix_undistorted.at<double>(1, 2) = 240;
    // camMatrix_undistorted.at<double>(2, 2) = 1;

    // galaxy 720p, e.g. from arcore pose recorder app

    camMatrix_raw.at<double>(0, 0) = 1126.3 / 2;
    camMatrix_raw.at<double>(1, 1) = 1165.46 / 2;
    camMatrix_raw.at<double>(0, 2) = 709.217 / 2;
    camMatrix_raw.at<double>(1, 2) = 356.259 / 2;
    camMatrix_raw.at<double>(2, 2) = 1;

    camMatrix_undistorted.at<double>(0, 0) = 1140 / 2;
    camMatrix_undistorted.at<double>(1, 1) = 1140 / 2;
    camMatrix_undistorted.at<double>(0, 2) = 1440/4;  // divide by 4 for 360p
    camMatrix_undistorted.at<double>(1, 2) = 720/4;
    camMatrix_undistorted.at<double>(2, 2) = 1;


    long lastImgTimestamp = 0;
    long lastARCTimestamp = 0;
    long currentARCTimestamp = 0;
    //    localizer = std::shared_ptr<pca::ACSLocalizer>(new pca::ACSLocalizer());

 //   cloudMapper = std::make_shared<PointCloudMapping>(0.05f);

    std::cout << "point cloud viewer created" << std::endl;
    
    std::string bundle_file("/home/demouser/Documents/RTAB-Map/bundler/office_va8_medium/cameras.out"); // TODO: change path to the bundler file exported by rtabmap
    std::string vw_assignments("/home/demouser/Documents/RTAB-Map/bundler/office_va8_medium/bundle.desc_assignments.integer_mean.voctree.clusters.100k.bin"); // TODO: change path to the bundler file exported by descriptor assignment tool
    std::string cluster_file("/home/demouser/acg_localizer/ACG-localizer/markt_paris_gpu_sift_100k.cluster");    // not model-specific


    loadPointCloudFromBIN(vw_assignments, cloudMapper, 100000);
//cloudMapper->AddPointCloud("cloud.ply");    // TODO: change path to the actual point cloud    
  //  cloudMapper->AddTexturedPolygonFromOBJ("/media/mikhail/0FD2D686317633C0/Datasets/rtabmap_DBs/office/december/mesh.obj");
    localizer.init(bundle_file, 100000, cluster_file, vw_assignments, 0, 500, 10);
    std::cout << "starting localizer loop" << std::endl;
    cv::Mat tempARCTransform = cv::Mat::eye(4, 4, CV_64F);
    //  uint64 tempARCTimestamp = 0;
    uint64 tempImgTimestamp = 0;
    //   np->Start();
    int imgIndex = 1;
    ofstream posesFile;
    posesFile.open ("/home/demouser/Documents/RTAB-Map/va_office/test_video/poses-17521fa0bbf.txt");
    cv::VideoCapture cap("/home/demouser/Documents/RTAB-Map/va_office/test_video/video-17521fa0ae5.mp4"); // Change video input path here
    int i = 0;
    while (true)
    {
       // logger.debug("waiting for request");
        
        cv::Mat frame; // bw input image 
        cap >> frame;
	cv::Size size1(775, 360);
        cv::resize(frame, frame, size1);
        if (frame.empty()) {
            std::cerr << "frame empty" << std::endl;
            break;
        }

        cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
        //frame = frame.reshape(1, 480);
        //   cv::Mat channels[3];
        //   cv::split(frame, channels);
        //           ms.debug() << "image channels: " << frame.channels() << std::endl;
        //           ms.debug() << "image width: " << frame.cols << " , height: " << frame.rows << std::endl;

        //
    //    logger.debug("frame.rows > 0");
        cv::undistort(frame, cameraImage, camMatrix_raw, cv::noArray());
    //    frame.copyTo(cameraImage);
        cv::imshow("Query Image", cameraImage);

     //   cv::imwrite("img/"std::to_string(imgIndex) + ".jpg", cameraImage);
        cv::waitKey(1);
        imgIndex++;
     //   std::cout << "localizer processing" << std::endl;
        cv::Mat inliers;
        std::vector<float> c2D, c3D;
        cv::Mat mDescriptors_q;
        std::set<size_t> unique_vw;
        //   arcoreTransform.copyTo(tempARCTransform);
        //    tempARCTimestamp = lastARCTimestamp;
        tempImgTimestamp = lastImgTimestamp;
        cv::Mat tempTransform = localizer.processImage(cameraImage, camMatrix_undistorted, inliers, c2D, c3D, mDescriptors_q, unique_vw);
        i++;
        uint32_t nb_corr = c2D.size() / 2;
        std::cout << "localizer found " <<  inliers.rows << " inliers " << " and correspondences: " << nb_corr << endl;
        if (tempTransform.empty() || inliers.rows <= 20)
        {
            // send response_msg, that pose could not be computed
            continue;
        }
       // std::cout << tempTransform << std::endl;
        posesFile.precision(12);    // 12 digit precision for pose values
        posesFile << i-1 << " " << inliers.rows << " " << 
        tempTransform.at<double>(0, 0) << " " << tempTransform.at<double>(0, 1) << " " << tempTransform.at<double>(0, 2) << " " << tempTransform.at<double>(0, 3) << " "  <<
        tempTransform.at<double>(1, 0) << " " << tempTransform.at<double>(1, 1) << " " << tempTransform.at<double>(1, 2) << " " << tempTransform.at<double>(1, 3) << " "  <<
        tempTransform.at<double>(2, 0) << " " << tempTransform.at<double>(2, 1) << " " << tempTransform.at<double>(2, 2) << " " << tempTransform.at<double>(2, 3) << " "  <<
        tempTransform.at<double>(3, 0) << " " << tempTransform.at<double>(3, 1) << " " << tempTransform.at<double>(3, 2) << " " << tempTransform.at<double>(3, 3) << " "  << 
        std::endl;
        
        cv::Mat sceneTransform = tempTransform.inv();    // inverting the transform since the client is interested in the scene pose
        acsTransform = tempTransform;
        cloudMapper.AddOrUpdateFrustum("2", acsTransform.inv(), 0.5, 0, 0, 1, 3, false, acs_opticalRotInv, 0.5, 0.5);
        // arcoreSnapshot should be the inverse of the arcore pose closest in time to the image pose timestamp.
        localizerWorking = true;
        poseValid = true;

        if (cameraImage.data)
        {
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
            // str << "D(TS): " <<  tempImgTimestamp - tempARCTimestamp : tempARCTimestamp - tempImgTimestamp;
            // cv::putText(colorImage,
            //             str.str(),
            //             cv::Point(5, 61),
            //             font,
            //             textSize,
            //             cv::Scalar(0, 255, 0));
            // str.str("");
            // str.clear();
            // str << "ARC TS: " << tempARCTimestamp;
            // cv::putText(colorImage,
            //             str.str(),
            //             cv::Point(5, 73),
            //             font,
            //             textSize,
            //             cv::Scalar(0, 255, 0));
            for (int i = 0; i < c2D.size(); i += 2)
            {
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::circle(colorImage, cv::Point(c2D[i] + (colorImage.cols - 1.0) / 2.0f, -c2D[i + 1] + (colorImage.rows - 1.0) / 2.0f), 4.0, color);
            }
            cv::imshow("Most Recent Localization", colorImage);
            //std::cout << "img timestamps: " << std::endl;
            //for (auto i = imgTimestamps.begin(); i != imgTimestamps.end(); ++i)
            //    std::cout << *i << ' ';
            //   std::cout << "img timestamp: " << tempImgTimestamp - timestampOffset << std::endl;
            //   std::cout << "pose timestamp: " << tempARCTimestamp - timestampOffset << std::endl;
            // std::cout << "pose timestamps: " << std::endl;
            // for (auto i = poseTimestamps.begin(); i != poseTimestamps.end(); ++i) {
            //     std::cout << *i - timestampOffset << ' ';
            // }
            // std::cout << std::endl;

            //   std::cout << "closest pose to img: " << indClosest << ", which is " << poseTimestamps[indClosest] - timestampOffset << std::endl;
            // std::cout << "closest pose to img: " << poseTimestamps[closest(poseTimestamps, tempImgTimestamp)] << std::endl;

            cv::waitKey(1);
        }
        localizerWorking = false;
    }
    posesFile.close();
    cout << "Press any key to quit.." << endl;
    // press Enter in the console to quit
    std::cin.ignore();
    
 //   logger.information("Stopping dealer");
    // dealer->Stop();
    //
  //  logger.information("Stopping proxy");
    // proxy->Stop();
    return 0;
}
