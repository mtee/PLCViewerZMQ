#include "pointcloudmapping.h"
#include "positioning.pb.h"

#include "acs_localizer.h"

#include "opencv2/core/eigen.hpp"
#include "opencv2/plot.hpp"
#include "opencv2/core.hpp"

#include <zmq.hpp>

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std::chrono;

const size_t width = 300;
const size_t height = 300;

cv::Mat acs_opticalRotInv = (cv::Mat_<double>(4,4) <<
            0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1);
cv::Mat arcore_opticalRotInv = cv::Mat::eye(4, 4, CV_64F);

std::string LOGTAG = "ZMQViewer";

cv::Mat camMatrix_raw = cv::Mat::eye(3, 3, CV_64F);
cv::Mat camMatrix_undistorted = cv::Mat::eye(3, 3, CV_64F);
cv::Mat cameraImage;
cv::Mat colorImage;
cv::Mat arcoreTransform = cv::Mat::eye(4, 4, CV_64F);
cv::Mat arcoreDiff = cv::Mat::eye(4, 4, CV_64F);
cv::Mat arcoreSnapshot = cv::Mat::eye(4, 4, CV_64F); // snapshot of the arcore pose valid at the moment of the most recent localization
cv::Mat acsTransform = cv::Mat::eye(4, 4, CV_64F);

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

// void receive(PBMessage msg)
// {
//     if (msg.has_pose())
//     {
//     //    ms.debug() << "got message with pose" << std::endl;
//         //    ms.information() << std::to_string(msg.pose().translation().x()) << " " << std::to_string(msg.pose().translation().y()) << " " << std::to_string(msg.pose().translation().z()) << std::endl;
//         //    ms.information() << std::to_string(msg.pose().rotation().x()) << " " << std::to_string(msg.pose().rotation().y()) << " " << std::to_string(msg.pose().rotation().z()) << " " << std::to_string(msg.pose().rotation().w()) << std::endl;
//         Eigen::Quaterniond q;
//         q.x() = msg.pose().rotation().x();
//         q.y() = msg.pose().rotation().y();
//         q.z() = msg.pose().rotation().z();
//         q.w() = msg.pose().rotation().w();

//         Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);

//         euler(0) = -euler(0);
//         euler(2) = -euler(2);

//         q = Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ());

//         Eigen::Matrix3d R = q.normalized().toRotationMatrix();
//         cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
//         for (int i = 0; i < 3; i++)
//             for (int j = 0; j < 3; j++)
//                 trans(i, j) = R(i, j);
//         trans(0, 3) = msg.pose().translation().x();
//         trans(1, 3) = -msg.pose().translation().y();
//         trans(2, 3) = msg.pose().translation().z();
//         //trans.copyTo(arcoreTransform);
//         if (timestampOffset == 0)
//         {
//             // Subtract first timestamp from all following timestamps
//             timestampOffset = msg.timestamp();
//         }
//         lastARCTimestamp = msg.timestamp();
//         if (!localizerWorking)
//         {
//             poseTimestamps.insert(poseTimestamps.end(), lastARCTimestamp);
//             poses.insert(poses.end(), trans);
//             if (poseTimestamps.size() > posesTotal)
//             {
//                 pop_front(poseTimestamps);
//                 pop_front(poses);
//             }
//         }
//             cloudMapper.AddOrUpdateFrustum("3", (acsTransform * (arcoreSnapshot * trans)).inv(), 0.5, 0, 1, 0.5, 4, false, acs_opticalRotInv, 0.5, 0.5);
//     }
// }

int main(int argc, const char *argv[])
{
    pca::ACSLocalizer localizer;
    poseValid = false;
    camMatrix_raw.at<double>(0, 0) = 489.776;
    camMatrix_raw.at<double>(1, 1) = 493.376;
    camMatrix_raw.at<double>(0, 2) = 316.348;
    camMatrix_raw.at<double>(1, 2) = 238.132;
    camMatrix_raw.at<double>(2, 2) = 1;

    camMatrix_undistorted.at<double>(0, 0) = 490;
    camMatrix_undistorted.at<double>(1, 1) = 490;
    camMatrix_undistorted.at<double>(0, 2) = 320;
    camMatrix_undistorted.at<double>(1, 2) = 240;
    camMatrix_undistorted.at<double>(2, 2) = 1;

    lastImgTimestamp = 0;
    lastARCTimestamp = 0;
    currentARCTimestamp = 0;
    PointCloudMapping cloudMapper(0.05f);
    std::string bundle_file("/home/demouser/Documents/RTAB-Map/bundler/office_va9_large/cameras.out"); // TODO: change path to the bundler file exported by rtabmap
    std::string vw_assignments("/home/demouser/Documents/RTAB-Map/bundler/office_va9_large/bundle.desc_assignments.integer_mean.voctree.clusters.100k.bin"); // TODO: change path to the bundler file exported by descriptor assignment tool
    std::string cluster_file("/home/demouser/acg_localizer/ACG-localizer/markt_paris_gpu_sift_100k.cluster");    // not model-specific


    cloudMapper.AddPointCloud("/home/demouser/Documents/RTAB-Map/bundler/office_va9_large/cloud.ply");
    localizer.init(bundle_file, 100000, cluster_file, vw_assignments, 0, 500, 10);
    std::cout << "starting localizer loop" << std::endl;
    cv::Mat tempARCTransform = cv::Mat::eye(4, 4, CV_64F);
    //  uint64 tempARCTimestamp = 0;
    uint64 tempImgTimestamp = 0;
    //   np->Start();
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:8081");
    int imgIndex = 1;
    while (true)
    {
        std::cout << "waiting for request" << std::endl;
        zmq::message_t request;
        socket.recv(&request);
        proto::positioning::Query msg;
        msg.ParseFromArray(request.data(), request.size());
        std::cout << "Received " << request.size() << "bytes" << std::endl;
        uint64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        proto::positioning::VisualMeasurement vm = msg.visual_measurements(0);
        std::string imageStr = vm.picture();
        std::cout << "got image sample" << std::endl;

        std::vector<uchar> imgBuf(imageStr.begin(), imageStr.end());
        cv::Mat frame = cv::Mat(imgBuf, true);
        //  cv::resize(frame, frame, size1);
        frame = frame.reshape(1, 480);  // assuming 480p resolution!
        if (frame.rows < 1)
        {
            // no valid image
            // TODO: perhaps send an error response back?
            continue;
        }
        cv::undistort(frame, cameraImage, camMatrix_raw, cv::noArray());
        cv::imshow("Query Image", cameraImage);
        cv::waitKey(1);
        imgIndex++;
        std::cout << "localizer processing" << std::endl;
        cv::Mat inliers;
        std::vector<float> c2D, c3D;
        cv::Mat mDescriptors_q;
        std::set<size_t> unique_vw;
        //   arcoreTransform.copyTo(tempARCTransform);
        //    tempARCTimestamp = lastARCTimestamp;
        cv::Mat tempTransform = localizer.processImage(cameraImage, camMatrix_undistorted, inliers, c2D, c3D, mDescriptors_q, unique_vw);
        
        if (tempTransform.empty() || inliers.rows <= 5)
        {
            // TODO: send response, that pose could not be computed
            continue;
        }
        cv::Mat sceneTransform = tempTransform.inv();    // inverting the transform since the client is interested in the scene pose
        std::cout << tempTransform << std::endl;
        std::cout << "scene: " << std::endl;
        std::cout << sceneTransform << std::endl;
        std::cout << "no image sample" << std::endl;
        localizerWorking = true;
        // Convert cv::mat tempTransform to quaternion and translation vector
        Eigen::Matrix3d m;
        cv::cv2eigen(sceneTransform(cv::Range(0, 3), cv::Range(0, 3)), m);
        Eigen::Quaterniond q(m);
        std::cout << "no image sample" << std::endl;
        
        proto::positioning::Response responsePose;
        // TODO: set pose in the response object and write it into tmp
        std:: string tmp = "";    // response_msg->SerializeAsString()
        zmq::message_t response(tmp.size());
        memcpy(response.data(), tmp.data(), tmp.size());
        socket.send(response);
        acsTransform = tempTransform;
        cloudMapper.AddOrUpdateFrustum("2", acsTransform.inv(), 0.5, 0, 0, 1, 3, false, acs_opticalRotInv, 0.5, 0.5);
        // arcoreSnapshot should be the inverse of the arcore pose closest in time to the image pose timestamp.
        localizerWorking = true;
        if (!poses.empty()) {
            arcoreSnapshot = poses[closest(poseTimestamps, tempImgTimestamp)].inv();
        }
        poseValid = true;
        uint32_t nb_corr = c2D.size() / 2;
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
            for (int i = 0; i < c2D.size(); i += 2)
            {
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::circle(colorImage, cv::Point(c2D[i] + (colorImage.cols - 1.0) / 2.0f, -c2D[i + 1] + (colorImage.rows - 1.0) / 2.0f), 4.0, color);
            }
            cv::imshow("Query Image", colorImage);
            cv::waitKey(1);
        }
        localizerWorking = false;
    }

    cout << "Press any key to quit.." << endl;
    // press Enter in the console to quit
    std::cin.ignore();
    return 0;
}
