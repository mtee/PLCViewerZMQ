#include <iostream>
#include <fstream>
#include <chrono>

#include "opencv2/core.hpp"

#include <Poco/SharedPtr.h>
#include <Poco/Logger.h>
#include <Poco/LogStream.h>

#include "acs_localizer.h"

#include "Poco/Thread.h"

#include "pointcloudmapping.h"
#include "opencv2/core/eigen.hpp"

using namespace Poco;
//using namespace cv;
using namespace std::chrono;

const size_t width = 300;
const size_t height = 300;

Poco::SharedPtr<PointCloudMapping> cloudMapper;
std::string LOGTAG = "ZMQViewer";


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
cv::Mat arcore_opticalRotInv = cv::Mat::eye(4, 4, CV_64F);

int lastACSLocalizedFrame = 0;
cv::Mat lastACSTransform;

bool poseValid;
bool localizerWorking;
cv::RNG rng(12345);
// substract 1548688800000 to get seconds
uint64 timestampOffset = 0;

int posesTotal = 24;

uint64 lastImgTimestamp;
uint64 lastARCTimestamp;
uint64 currentARCTimestamp;

//std::vector<uint64> imgTimestamps;
std::vector<uint64> poseTimestamps;
std::vector<cv::Mat> poses;

int counter = 0;

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

cv::Size size1(640, 480);

void readArcorePoses(std::string fileName, std::vector<std::string> &imgFiles, std::vector<cv::Mat> &poses) {
    std::ifstream arcoreSessionFile(fileName);
    if(arcoreSessionFile.is_open()) {
        std::cout << "File opened: " << fileName << std::endl;
        std::string line;
        std::vector<std::string> tokens;

        int nLine = 0;
        int nVideoFrame = 0;
        std::string folderPath = fileName.substr(0, fileName.find_last_of('/'));
        std::cout << folderPath << std::endl;
        while (std::getline(arcoreSessionFile, line))
        {
            if (line.length() < 1)
                continue;
            tokens = split(line);
            int imgIndex = std::stoi( tokens[0] ); // the frame ids begin from 0, however the image filenames begin with 1.
            // so the frame #0 has the image path 1.png
            imgFiles.push_back(folderPath + "/frames/" + std::to_string(imgIndex) + ".png");
            cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);            
            double x = toDouble(tokens[1]);
            double y = toDouble(tokens[2]);
            double z = toDouble(tokens[3]);
            Eigen::Quaterniond q;
            q.x() = toDouble(tokens[4]);
            q.y() = toDouble(tokens[5]);
            q.z() = toDouble(tokens[6]);
            q.w() = toDouble(tokens[7]);

            Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
            //TODO: figure out the right conversion here for euler angles           
            
            q = Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX())       
            * Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) 
            * Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ());

            Eigen::Matrix3d R = q.normalized().toRotationMatrix();
            
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    trans(i, j) = R(i, j);
            

            trans(0, 3) = x;      // TODO: perhaps we have to switch coordinates here as well
            trans(1, 3) = y;
            trans(2, 3) = z;
            poses.push_back(trans);
        }
        
    }
}

int main(int argc, const char *argv[])
{
    std::vector<std::string> imgFiles; 
    std::vector<cv::Mat> poses;
    std::string arsession("/media/mikhail/0FD2D686317633C0/Datasets/arcore_sessions/office/calib_2/arposes.txt");
    readArcorePoses(arsession, imgFiles, poses);
    std::cout << "poses total: " << poses.size() << ". Images total: " << imgFiles.size() << std::endl;

    pca::ACSLocalizer localizer;
    poseValid = false;
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
    camMatrix_undistorted.at<double>(0, 2) = 1440 / 4;
    camMatrix_undistorted.at<double>(1, 2) = 720 / 4;
    camMatrix_undistorted.at<double>(2, 2) = 1;


    lastImgTimestamp = 0;
    lastARCTimestamp = 0;
    currentARCTimestamp = 0;
    //    localizer = Poco::SharedPtr<pca::ACSLocalizer>(new pca::ACSLocalizer());

    Logger &logger = Logger::get(LOGTAG);

    cloudMapper = Poco::SharedPtr<PointCloudMapping>(new PointCloudMapping(0.05f));

    std::cout << "point cloud viewer created" << std::endl;

    //  cloudMapper->AddTexturedPolygonFromOBJ("/media/mikhail/0FD2D686317633C0/Datasets/rtabmap_DBs/office/december/mesh.obj");

    std::string dataFolder("/media/mikhail/0FD2D686317633C0/Datasets/rtabmap_DBs/office/december");
    std::string cloudFilename("cloud.ply");
    
    
    
    std::string bundle_file(dataFolder + "/bundler/cameras.out"); // TODO: change path to the bundler file exported by rtabmap
    std::string vw_assignments(dataFolder + "/bundler/desc_assignments.integer_mean.voctree.clusters.bin"); // TODO: change path to the bundler file exported by descriptor assignment tool
    std::string cluster_file("/home/mikhail/workspace/ACG-localizer/markt_paris_gpu_sift_100k.cluster");    // not model-specific


    cloudMapper->AddPointCloud(dataFolder + "/" + cloudFilename);
    localizer.init(bundle_file, 100000, cluster_file, vw_assignments, 0, 500, 10);
    std::cout << "starting localizer loop" << std::endl;
    cv::Mat tempARCTransform = cv::Mat::eye(4, 4, CV_64F);
    //  uint64 tempARCTimestamp = 0;
    uint64 tempImgTimestamp = 0;
    //   np->Start();
    for (int i = 0; i < imgFiles.size(); i++)
    {
        logger.debug("waiting for request");
        
        cv::Mat frame = imread(imgFiles[i], cv::IMREAD_ANYCOLOR); // bw input image 
        if (frame.empty()) {
            std::cout << "image is empty: " << imgFiles[i] << std::endl;
            continue;
        }
        //  cv::resize(frame, frame, size1);
        cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
        //frame = frame.reshape(1, 480);
        //   cv::Mat channels[3];
        //   cv::split(frame, channels);
        //           ms.debug() << "image channels: " << frame.channels() << std::endl;
        //           ms.debug() << "image width: " << frame.cols << " , height: " << frame.rows << std::endl;

        //
        logger.debug("frame.rows > 0");
        cv::undistort(frame, cameraImage, camMatrix_raw, cv::noArray());
    //    frame.copyTo(cameraImage);
        cv::imshow("Query Image", cameraImage);

     //   cv::imwrite("img/"std::to_string(imgIndex) + ".jpg", cameraImage);
        cv::waitKey(1);
     //   std::cout << "localizer processing" << std::endl;
        cv::Mat inliers;
        std::vector<float> c2D, c3D;
        cv::Mat mDescriptors_q;
        std::set<size_t> unique_vw;
        //   arcoreTransform.copyTo(tempARCTransform);
        //    tempARCTimestamp = lastARCTimestamp;
        tempImgTimestamp = lastImgTimestamp;
        
        cv::Mat tempTransform = localizer.processImage(cameraImage, camMatrix_undistorted, inliers, c2D, c3D, mDescriptors_q, unique_vw);
        tempTransform = cv::Mat::eye(4, 4, CV_64F);
        std::cout << "localizer found " <<  inliers.rows << " inliers" << std::endl;
        if (tempTransform.empty() || i >= 1) {
            if (lastACSTransform.empty())
                lastACSTransform = cv::Mat::eye(4, 4, CV_64F);
            cv::Mat delta = poses[lastACSLocalizedFrame].inv() * poses[i];
            acsTransform = lastACSTransform * delta;
            cloudMapper->AddOrUpdateFrustum("1", acsTransform.inv(), 0.5, 0, 1, 0, 3, false, acs_opticalRotInv, 0.5, 0.5);
            continue;
        }
        if (inliers.rows <= 20)
        {
            // send response_msg, that pose could not be computed
            continue;
        }
        lastACSLocalizedFrame = i;
        lastACSTransform = tempTransform;
        acsTransform = tempTransform * poses[lastACSLocalizedFrame].inv() * poses[i];
        
        
        
        cloudMapper->AddOrUpdateFrustum("2", acsTransform.inv(), 0.5, 0, 0, 1, 3, false, acs_opticalRotInv, 0.5, 0.5);
        logger.debug("AddOrUpdateFrustum");
        // arcoreSnapshot should be the inverse of the arcore pose closest in time to the image pose timestamp.

        localizerWorking = true;
        poseValid = true;
        uint32_t nb_corr = c2D.size() / 2;
        logger.debug("correspondences");
        if (cameraImage.data)
        {
            logger.debug("cameraImage.data");
            cv::cvtColor(cameraImage, colorImage, CV_GRAY2BGR);
            logger.debug("colorImage.data");
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
    cout << "Press any key to quit.." << endl;
    // press Enter in the console to quit
    std::cin.ignore();
    logger.information("Stopping dealer");
    // dealer->Stop();
    //
    logger.information("Stopping proxy");
    // proxy->Stop();
    return 0;
}
