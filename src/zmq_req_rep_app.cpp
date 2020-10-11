#include <iostream>
#include <fstream>
#include <chrono>

#include "opencv2/plot.hpp"
#include "opencv2/core.hpp"

#include <Poco/Logger.h>
#include <Poco/LogStream.h>

#include <zmq.hpp>

#include <Dealer.h>
#include <NetworkPublisher.h>
#include <MessageProxy.h>
#include <attentive_framework.pb.h>

#include "acs_localizer.h"

#include "pointcloudmapping.h"
#include "opencv2/core/eigen.hpp"

using namespace attentivemachines;
using namespace Poco;
//using namespace cv;
using namespace std::chrono;

const size_t width = 300;
const size_t height = 300;

Poco::SharedPtr<PointCloudMapping> cloudMapper;
cv::Mat acs_opticalRotInv = (cv::Mat_<double>(4,4) <<
            0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1);
cv::Mat arcore_opticalRotInv = cv::Mat::eye(4, 4, CV_64F);
//Poco::SharedPtr<pca::ACSLocalizer> localizer;

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

PBImageSample imgSample;
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

void receive(PBMessage msg)
{
    Logger &logger = Logger::get(LOGTAG);
    Poco::LogStream ms(logger);
    if (msg.has_pose())
    {
    //    ms.debug() << "got message with pose" << std::endl;
        //    ms.information() << std::to_string(msg.pose().translation().x()) << " " << std::to_string(msg.pose().translation().y()) << " " << std::to_string(msg.pose().translation().z()) << std::endl;
        //    ms.information() << std::to_string(msg.pose().rotation().x()) << " " << std::to_string(msg.pose().rotation().y()) << " " << std::to_string(msg.pose().rotation().z()) << " " << std::to_string(msg.pose().rotation().w()) << std::endl;
        Eigen::Quaterniond q;
        q.x() = msg.pose().rotation().x();
        q.y() = msg.pose().rotation().y();
        q.z() = msg.pose().rotation().z();
        q.w() = msg.pose().rotation().w();

        Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);

        euler(0) = -euler(0);
        euler(2) = -euler(2);

        q = Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ());

        Eigen::Matrix3d R = q.normalized().toRotationMatrix();
        cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                trans(i, j) = R(i, j);
        trans(0, 3) = msg.pose().translation().x();
        trans(1, 3) = -msg.pose().translation().y();
        trans(2, 3) = msg.pose().translation().z();
        //trans.copyTo(arcoreTransform);
        if (timestampOffset == 0)
        {
            // Subtract first timestamp from all following timestamps
            timestampOffset = msg.timestamp();
        }
        lastARCTimestamp = msg.timestamp();
        if (!localizerWorking)
        {
            poseTimestamps.insert(poseTimestamps.end(), lastARCTimestamp);
            poses.insert(poses.end(), trans);
            if (poseTimestamps.size() > posesTotal)
            {
                pop_front(poseTimestamps);
                pop_front(poses);
            }
        }
            cloudMapper->AddOrUpdateFrustum("3", (acsTransform * (arcoreSnapshot * trans)).inv(), 0.5, 0, 1, 0.5, 4, false, acs_opticalRotInv, 0.5, 0.5);
    }
}

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
    //    localizer = Poco::SharedPtr<pca::ACSLocalizer>(new pca::ACSLocalizer());

    Config *config = CommonsTool::getConfig("bin/ComExample.ini");
    config->type = "localizer";
    config->publishTypes.push_back(PBMessage::kPose);
    CommonsTool::InitLogging(config);
    Logger &logger = Logger::get(LOGTAG);

 //   Poco::SharedPtr<NetworkPublisher> np = Poco::SharedPtr<NetworkPublisher>(new NetworkPublisher());
 //   np->Initialize(config);

    Poco::SharedPtr<Dealer> dealer = Poco::SharedPtr<Dealer>(new Dealer());
    dealer->Initialize(config);
    Poco::SharedPtr<MessageProxy> proxy = Poco::SharedPtr<MessageProxy>(new MessageProxy());
    std::list<std::string> devices;
  //  devices.push_back("img");
    devices.push_back("pose");
    proxy->Initialize(std::list<std::string>(), devices, dealer);
    proxy->AddSubscriber(receive);
    dealer->Start();
    proxy->Start();

    std::cout << "hello pointcloud" << std::endl;

    cloudMapper = Poco::SharedPtr<PointCloudMapping>(new PointCloudMapping(0.05f));

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
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:4004");
    int imgIndex = 1;
    while (true)
    {
        logger.debug("waiting for request");
        zmq::message_t request;
        socket.recv(&request);
        PBMessage msg;
        msg.ParseFromArray(request.data(), request.size());
        std::cout << "Received " << request.size() << "bytes" << std::endl;
        uint64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        PBMessage* response_msg = new PBMessage();
        response_msg->set_source(165432);
        response_msg->set_target(0);
        response_msg->set_timestamp(now);
        if (!msg.has_image())
        {
            logger.debug("no image sample");
            PBErrorResponse* err = new PBErrorResponse();
            err->set_msg("noimg");
            response_msg->set_allocated_error(err);
            std:: string tmp = response_msg->SerializeAsString();
            zmq::message_t response(tmp.size());
            memcpy(response.data(), tmp.data(), tmp.size());
            socket.send(response);
            continue;
        }
        imgSample = msg.image();
        logger.debug("got message with image sample");
        if (imgSample.compression() != PBImageSample_Compression_JPG)
        {
            PBErrorResponse* err = new PBErrorResponse();
            err->set_msg("nojpg");
            response_msg->set_allocated_error(err);
            std:: string tmp = response_msg->SerializeAsString();
            zmq::message_t response(tmp.size());
            memcpy(response.data(), tmp.data(), tmp.size());
            socket.send(response);
            continue;
        }
        logger.debug("Image sample has JPG compression");
        //    time_.Start();
        if (imgSample.img().empty())
        {
            PBErrorResponse* err = new PBErrorResponse();
            err->set_msg("emptyimg");
            response_msg->set_allocated_error(err);
            std:: string tmp = response_msg->SerializeAsString();
            zmq::message_t response(tmp.size());
            memcpy(response.data(), tmp.data(), tmp.size());
            socket.send(response);
            continue;
        }
        logger.debug("!imgString.empty()");
        std::vector<uchar> imgBuf(imgSample.img().begin(), imgSample.img().end());
        //   cv::Mat frame = cv::imdecode(imgBuf, cv::IMREAD_UNCHANGED);
        cv::Mat frame = cv::Mat(imgBuf, true);
        //  cv::resize(frame, frame, size1);
        frame = frame.reshape(1, 480);
        //   cv::Mat channels[3];
        //   cv::split(frame, channels);
        //           ms.debug() << "image channels: " << frame.channels() << std::endl;
        //           ms.debug() << "image width: " << frame.cols << " , height: " << frame.rows << std::endl;

        //
        if (frame.rows < 1)
        {
            PBErrorResponse* err = new PBErrorResponse();
            err->set_msg("emptyimg");
            response_msg->set_allocated_error(err);
            std:: string tmp = response_msg->SerializeAsString();
            zmq::message_t response(tmp.size());
            memcpy(response.data(), tmp.data(), tmp.size());
            socket.send(response);
            continue;
        }
        logger.debug("frame.rows > 0");
        cv::undistort(frame, cameraImage, camMatrix_raw, cv::noArray());
    //    frame.copyTo(cameraImage);
        cv::imshow("Query Image", cameraImage);

     //   cv::imwrite("img/"std::to_string(imgIndex) + ".jpg", cameraImage);
        cv::waitKey(1);
        imgIndex++;
        lastImgTimestamp = msg.timestamp();
        std::cout << "localizer processing" << std::endl;
        cv::Mat inliers;
        std::vector<float> c2D, c3D;
        cv::Mat mDescriptors_q;
        std::set<size_t> unique_vw;
        //   arcoreTransform.copyTo(tempARCTransform);
        //    tempARCTimestamp = lastARCTimestamp;
        tempImgTimestamp = lastImgTimestamp;
        cv::Mat tempTransform = localizer.processImage(cameraImage, camMatrix_undistorted, inliers, c2D, c3D, mDescriptors_q, unique_vw);
        
        if (tempTransform.empty() || inliers.rows <= 20)
        {
            // send response_msg, that pose could not be computed

            PBErrorResponse* err = new PBErrorResponse();
            err->set_msg("nopose");
            response_msg->set_allocated_error(err);
            std:: string tmp = response_msg->SerializeAsString();
            zmq::message_t response(tmp.size());
            memcpy(response.data(), tmp.data(), tmp.size());
            socket.send(response);
            continue;
        }
        cv::Mat sceneTransform = tempTransform.inv();    // inverting the transform since the client is interested in the scene pose
        std::cout << tempTransform << std::endl;
        std::cout << "scene: " << std::endl;
        std::cout << sceneTransform << std::endl;
        logger.debug("processed");
        localizerWorking = true;
        logger.debug("enough inliers");
        // Convert cv::mat tempTransform to quaternion and translation vector
        Eigen::Matrix3d m;
        cv::cv2eigen(sceneTransform(cv::Range(0, 3), cv::Range(0, 3)), m);
        Eigen::Quaterniond q(m);
        logger.debug("converted cv to eigen");
        PBPoseSample* pose = new PBPoseSample();
        attentivemachines::Point3D* translation = new attentivemachines::Point3D();
        translation->set_x(sceneTransform.at<double>(0, 3));
        translation->set_y(sceneTransform.at<double>(1, 3));
        translation->set_z(sceneTransform.at<double>(2, 3));
        logger.debug("set translation");
        std::cout << "x: " << translation->x() << " " << translation->y() << " " << translation->z() << std::endl;
        // TODO check if x, y, z are set
        pose->set_allocated_translation(translation);
        attentivemachines::Quaternion* rotation = new attentivemachines::Quaternion();
        // TODO check x, y, z, w
        rotation->set_x(q.x());
        rotation->set_y(q.y());
        rotation->set_z(q.z());
        rotation->set_w(q.w());
        pose->set_allocated_rotation(rotation);
        logger.debug("set rotation");
        response_msg->set_allocated_pose(pose);
        std:: string tmp = response_msg->SerializeAsString();
        zmq::message_t response(tmp.size());
        memcpy(response.data(), tmp.data(), tmp.size());
        socket.send(response);
        logger.debug("response_msg sent");
        acsTransform = tempTransform;
        cloudMapper->AddOrUpdateFrustum("2", acsTransform.inv(), 0.5, 0, 0, 1, 3, false, acs_opticalRotInv, 0.5, 0.5);
        logger.debug("AddOrUpdateFrustum");
        // arcoreSnapshot should be the inverse of the arcore pose closest in time to the image pose timestamp.
        localizerWorking = true;
        if (!poses.empty()) {
            arcoreSnapshot = poses[closest(poseTimestamps, tempImgTimestamp)].inv();
            logger.debug("arcoreSnapshot");
        }
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
            cv::imshow("Query Image", colorImage);
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
