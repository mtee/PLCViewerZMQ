#include <iostream>
#include <fstream>

#include "pointcloudmapping.h"

#include "opencv2/plot.hpp"
#include <Poco/Logger.h>
#include <Poco/LogStream.h>
#include <zmq.hpp>
#include <Dealer.h>
#include <MessageProxy.h>
#include <attentive_framework.pb.h>


using namespace attentivemachines;
using namespace Poco;
//using namespace cv;
using namespace std::chrono;

const size_t width = 300;
const size_t height = 300;

Poco::SharedPtr<PointCloudMapping> cloudMapper;


std::string LOGTAG = "ZMQViewer";

PBImageSample imgSample;
std::string imgString;
int counter = 0;
std::ostringstream s;


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




 cv::Size size1(640, 480);

void receive(PBMessage msg) {
    Logger &logger = Logger::get(LOGTAG);
    Poco::LogStream ms(logger);
//    ms << "msg: " << std::to_string(msg);
    if (msg.has_image()) {
        imgSample = msg.image();
        logger.debug("got message with image sample");
        
        if (imgSample.compression() == PBImageSample_Compression_JPG) 
        {
            logger.debug("Image sample has JPG compression");
            //    time_.Start();
            imgString = imgSample.img();
            if (!imgString.empty())
            { 
                std::vector<uchar> imgBuf(imgString.begin(), imgString.end());
             //   cv::Mat frame = cv::imdecode(imgBuf, cv::IMREAD_UNCHANGED);
                cv::Mat frame = cv::Mat(imgBuf, true);
              //  cv::resize(frame, frame, size1);
                frame = frame.reshape(1, 480);
          
          
             //   cv::Mat channels[3];
             //   cv::split(frame, channels);
                ms.debug() << "image channels: " << frame.channels() << std::endl;
                ms.debug() << "image width: " << frame.cols << " , height: " << frame.rows << std::endl;

                //cv::imshow("Frame", channels[2]);
                if (frame.rows > 0) {
                    cv::imshow("Frame", frame);
                    cv::waitKey(3);
                } else {
                    ms.warning() << "image could not be decoded" << std::endl;
                }
            }
        } else if ((imgSample.compression() == PBImageSample_Compression_H264)) 
        {
            //    time_.Start();
            logger.warning("Image sample has h264 compression which is not implemented yet");
    //        std::vector<uchar> imgBuf(imgString.begin(), imgString.end());
       //     cv::Mat frame = cv::imdecode(imgBuf, 1);
      //      imshow("Frame", frame);
     //       waitKey(10);
        }
    }
    if (msg.has_pose()) {
        ms.debug() << "got message with pose" << std::endl;
        ms.information() << std::to_string(msg.pose().translation().x()) << " " << std::to_string(msg.pose().translation().y()) << " " << std::to_string(msg.pose().translation().z()) << std::endl;
        ms.information() << std::to_string(msg.pose().rotation().x()) << " " << std::to_string(msg.pose().rotation().y()) << " " << std::to_string(msg.pose().rotation().z()) << " " << std::to_string(msg.pose().rotation().w()) << std::endl;
        Eigen::Quaterniond q;
        q.x() = msg.pose().rotation().x();
        q.y() = msg.pose().rotation().y();
        q.z() = msg.pose().rotation().z();
        q.w() = msg.pose().rotation().w();
        
        Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);

        euler(0) = - euler(0);
        euler(2) = - euler(2);

        q = Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ());
        
        Eigen::Matrix3d R = q.normalized().toRotationMatrix();
        cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                trans(i,j) = R(i, j);
        trans(0, 3) = msg.pose().translation().x();
        trans(1, 3) = -msg.pose().translation().y();
        trans(2, 3) = msg.pose().translation().z();
        cloudMapper->AddOrUpdateFrustum("1", trans, 0.5, 1, 0, 0, 2);
    }
}


int main(int argc, const char* argv[])
{
    Config *config = CommonsTool::getConfig("bin/ComExample.ini");
    config->type = "pupil";
    CommonsTool::InitLogging(config);
    Logger &logger = Logger::get(LOGTAG);
    Poco::SharedPtr<Dealer> dealer = Poco::SharedPtr<Dealer>(new Dealer());
    dealer->Initialize(config);
    Poco::SharedPtr<MessageProxy> proxy = Poco::SharedPtr<MessageProxy>(new MessageProxy());
    std::list<std::string> devices;
    devices.push_back("pupil");
    proxy->Initialize(std::list<std::string>(), devices, dealer);
    proxy->AddSubscriber(receive);
    dealer->Start();
    proxy->Start();

    std::cout << "hello pointcloud" << std::endl;

    cloudMapper = Poco::SharedPtr<PointCloudMapping>(new PointCloudMapping(0.02f));
    std::cout << "point cloud viewer created" << std::endl;
    cloudMapper->AddPointCloud("bin/cloud.ply");
    std::ifstream posesFile("bin/poses317.txt");
    // if(posesFile.is_open()) {
    //     std::cout << "File opened" << std::endl;
    //     std::string line;
    //     std::vector<std::string> tokens;
    //     while (std::getline(posesFile, line))
    //     {
    //         if (line.length() < 1)
    //             continue;
    //         tokens = split(line);
    //         // Transformation
    //         cv::Mat pose;
    //         cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
    //         for(unsigned i = 0; i < 3; i++)
    //             for(unsigned j = 0; j < 4; j++)
    //             {
    //                 trans(i, j) =  toDouble(tokens[i*4 + j].c_str());
    //             }
    //         cloudMapper->AddOrUpdateFrustum("1", trans, 0.5, 1, 0, 0, 2);
    //         cv::waitKey(100);
    //     }
    // }
    cout << "Press any key to quit.." << endl;
    // press Enter in the console to quit
    std::cin.ignore();
    logger.information("Stopping dealer");
    dealer->Stop();
//
    logger.information("Stopping proxy");
    proxy->Stop();
    return 0;
}
