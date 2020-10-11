#include "heatmapexamplelauncher.h"

HeatMapExampleLauncher::HeatMapExampleLauncher(){}

inline double HeatMapExampleLauncher::toDouble(std::string s){
    std::replace(s.begin(), s.end(), ',', '.');
    return std::atof(s.c_str());
}

std::vector<std::string> HeatMapExampleLauncher::split(const std::string& s, char delim) 
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    
    while (std::getline(ss, item, delim)) elems.push_back(item);
    
    return elems;
}

std::vector<std::string> HeatMapExampleLauncher::split(const std::string& s) 
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
int HeatMapExampleLauncher::loadPointCloudFromBIN(std::string filename, PointCloudMapping& cloudMapper, uint32_t nb_clusters){
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

int HeatMapExampleLauncher::runCockpitExample(bool useSemanticGrid){

    std::cout << "Starting heat map visualization, cockpit example." << std::endl;
    PointCloudMapping cloudMapper(0.02f);           // 0.02f orig
    std::cout << "point cloud viewer created" << std::endl;

    cloudMapper.AddPointCloudFromOBJ("3dcockpit/untitled.obj");                  // pointcloud from 3D object

    cloudMapper.generateSearchOctree(0.1f, 20);
    
        // establish test semantic grid
    if(useSemanticGrid){
        cloudMapper.setSemanticMode(true);
        // cloudMapper.addToSemanticGrid("1", "Instrument #1", Eigen::Vector3f(2.18284, 1.2195, -1.20163), Eigen::Vector3f(2.38284, 1.4195, -1.00163));
        // cloudMapper.addToSemanticGrid("2", "Instrument #2", Eigen::Vector3f(2.08284, 0.8195, -1.20163), Eigen::Vector3f(2.28284, 1.0195, -1.00163));
        // cloudMapper.addToSemanticGrid("3", "Instrument #3", Eigen::Vector3f(2.48284, 0.8195, -1.20163), Eigen::Vector3f(2.68284, 1.0195, -1.00163));

        // test JSON reader
        cloudMapper.readSemanticGridFromJSON("cockpit.json");

        cloudMapper.addTitlesToSemanticGrid();
    }

    int nWaitSec = 15;
    std::cout << "Point cloud ready. Starting playback in " << std::endl;
    for(int i=nWaitSec; i>0; --i){
        std::cout << i << std::endl;
        cv::waitKey(1000);
    }

    std::ifstream posesFile("poses317double.txt");
    if(posesFile.is_open()) {
        std::cout << "File opened" << std::endl;
        std::string line;
        std::vector<std::string> tokens;

        int nLine = 0;
        int nVideoFrame = 0;

        while (std::getline(posesFile, line))
        {
            if (line.length() < 1)
                continue;
            tokens = split(line);
            // Transformation
            cv::Mat pose;
            cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
            for(unsigned i = 0; i < 3; i++)
                for(unsigned j = 0; j < 4; j++)
                {
                    trans(i, j) =  toDouble(tokens[i*4 + j].c_str()); // cockpit demo
                    
                    // scaling for cockpit demo
                    if(j==3) trans(i, j) *= 0.1;
                    else trans(i, j) *= 0.25;

                    // std::cout << "Frustrum [" << i << ", " << j << "] = " << trans(i,j) << std::endl;
                }

            // Translations for cockpit demo
            trans(0, 3) += 2.6;           // x translation
            trans(1, 3) += 1.8;           // y translation
            trans(2, 3) -= 2.6;           // z translation

            if(nLine>149){
            // trans(0, 3) += 0.1;           // x translation
            trans(1, 3) -= 0.15;           // y translation
            // trans(2, 3) += 0;           // z translation
            }

            // cloudMapper.AddOrUpdateFrustum("frustum", trans, 0.5, 1, 0, 0, 2);
            cloudMapper.AddOrUpdateFrustum("frustum", trans, 1, 1, 1, 1, 2, true, 0.5, 0.5, true);
            cv::waitKey(75);

        }
    }
    
    // press Enter in the console to quit
    std::cin.ignore();
    
    return 0;
}

int HeatMapExampleLauncher::runPCAOfficeExample(bool useSemanticGrid){
    std::cout << "Starting heat map visualization, PCA office example." << std::endl;
    PointCloudMapping cloudMapper(0.02f);           // 0.02f orig
    std::cout << "point cloud viewer created" << std::endl;

    if(loadPointCloudFromBIN("office.bin", cloudMapper, 100000)==-1) return -1;     // pointcloud from bin file (visual word assignment)
    // cloudMapper.AddPointCloud("cloud.ply");                                      // pointcloud from list of points
    // cloudMapper.AddPointCloudFromOBJ("3dcockpit/untitled.obj");                  // pointcloud from 3D object

    cloudMapper.generateSearchOctree(0.1f, 20);

    // establish test semantic grid
    if(useSemanticGrid){
        cloudMapper.setSemanticMode(true);
        // cloudMapper.addToSemanticGrid("1", "phone", Eigen::Vector3f(1.2434, -0.706912, -0.570855), Eigen::Vector3f(1.6434, -0.306912, -0.370855));
        // cloudMapper.addToSemanticGrid("2", "whiteboard", Eigen::Vector3f(-0.743402, -1.20691, -0.229145), Eigen::Vector3f(0.643402, -1.20691, 0.829145));
        // cloudMapper.addToSemanticGrid("3", "lanyard", Eigen::Vector3f(-1.943402, -1.10691, 0.129145), Eigen::Vector3f(-1.543402, -1.10691, 0.429145));

        // test JSON reader
        cloudMapper.readSemanticGridFromJSON("officetest.json");

        cloudMapper.addTitlesToSemanticGrid();
    }
    
    int nWaitSec = 5;
    std::cout << "Point cloud ready. Starting playback in " << std::endl;
    for(int i=nWaitSec; i>0; --i){
        std::cout << i << std::endl;
        cv::waitKey(1000);
    }

    // prepare gaze pos file
    double startTS = 89372.788099;
    double endTS = 89437.105219;

    int frameRange = 1275;
    double timePerFrame = (endTS - startTS) / frameRange;

    std::ifstream gazeFile("gaze_positions.csv");
    if(!gazeFile.is_open()){
        std::cout << "Error reading gaze file, aborting." << std::endl;

        return -1;
    }

    // prepare video output
    cv::VideoCapture cap("world.mp4");

    if(!cap.isOpened()){
        std::cout << "Error opening world video file, aborting." << std::endl;

        return -1;
    }

    // std::ifstream posesFile("poses317double.txt");
    std::ifstream posesFile("poses_office_2.txt");
    if(posesFile.is_open()) {
        std::cout << "File opened" << std::endl;
        std::string line;
        std::vector<std::string> tokens;

        int nLine = 0;
        int nVideoFrame = 0;

        cv::Mat_<double> currentLastTrans = cv::Mat_<double>::eye(4, 4);
        double currentLastTS = startTS;

        while (std::getline(posesFile, line))
        {
            if (line.length() < 1)
                continue;
            tokens = split(line);
            // Transformation
            cv::Mat pose;
            cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
            for(unsigned i = 0; i < 3; i++)
                for(unsigned j = 0; j < 4; j++)
                {
                    trans(i, j) =  toDouble(tokens[2 + i*4 + j].c_str()); // increm tokens i by 2 for office file
                }

            // visualize frustum with gaze positions before the current pose frame
            double gazeTS = 0.;
            double lastGazeNormX, lastGazeNormY;

            while(gazeTS < currentLastTS){
                std::string gazeLine;
                std::vector<std::string> gazeTokens;
                std::getline(gazeFile, gazeLine);

                gazeTokens = split(gazeLine, ',');

                gazeTS = toDouble(gazeTokens[0]);

                lastGazeNormX = toDouble(gazeTokens[3]);
                lastGazeNormY = toDouble(gazeTokens[4]);

                // std::cout << "On gazeTS " << gazeTS << ", currentLastTS " << currentLastTS << ". Got gaze norm pos [" << gazeNormX << ", " << gazeNormY << "]." << std::endl;
                // cloudMapper.AddOrUpdateFrustum("frustum", currentLastTrans, 1, 1, 1, 1, 2, true, lastGazeNormX, lastGazeNormY);
            }

            // only visualize with last gaze position
            if(gazeTS != 0.){
                cloudMapper.AddOrUpdateFrustum("frustum", currentLastTrans, 1, 1, 1, 1, 2, true, lastGazeNormX, lastGazeNormY);
            }

            // std::cout<< "Finished frame " << tokens[0] << "." << std::endl;
            // if(nLine>2) break;

            // visualize video until current frame
            cv::Mat videoFrame;

            while(nVideoFrame < toDouble(tokens[0])){
                cap >> videoFrame;
                cv::imshow("Frame", videoFrame);
                ++nVideoFrame;
            }

            // cloudMapper.AddOrUpdateFrustum("frustum", trans, 0.5, 1, 0, 0, 2);
            // cloudMapper.AddOrUpdateFrustum("frustum", trans, 1, 1, 1, 1, 2, true, 0., 0.);
            cv::waitKey(25);

            ++nLine;
            currentLastTrans = trans;
            currentLastTS = startTS + toDouble(tokens[0]) * timePerFrame;
        }
    }
    
    cap.release();

    // press Enter in the console to quit
    std::cin.ignore();

    return 0;
}

int HeatMapExampleLauncher::runVoestExample(bool useSemanticGrid){
    std::cout << "Starting heat map visualization, Voest example." << std::endl;
    PointCloudMapping cloudMapper(0.02f);           // 0.02f orig
    std::cout << "point cloud viewer created" << std::endl;

    if(loadPointCloudFromBIN("voest.bin", cloudMapper, 100000)==-1) return 0;      // pointcloud from bin file (visual word assignment)
    // cloudMapper.AddPointCloud("cloud_voest.ply");                                      // pointcloud from list of points

    // add textured polygon 
    cloudMapper.AddTexturedPolygonFromOBJ("voestmesh/mesh.obj");

    cloudMapper.generateSearchOctree(0.1f, 20);

    // establish semantic grid
    if(useSemanticGrid){
        cloudMapper.setSemanticMode(true);
        // cloudMapper.addToSemanticGrid("8", "phone", Eigen::Vector3f(1.2434, -0.706912, -0.570855), Eigen::Vector3f(1.6434, -0.306912, -0.370855));
        // cloudMapper.addToSemanticGrid("whiteboard", Eigen::Vector3f(-0.743402, -1.20691, -0.229145), Eigen::Vector3f(0.643402, -1.20691, 0.829145));
        // cloudMapper.addToSemanticGrid("lanyard", Eigen::Vector3f(-1.943402, -1.10691, 0.129145), Eigen::Vector3f(-1.543402, -1.10691, 0.429145));

        // JSON reader
        cloudMapper.readSemanticGridFromJSON("quickstart.json");

        cloudMapper.addTitlesToSemanticGrid();
    }

    int nWaitSec = 5;
    std::cout << "Point cloud ready. Starting playback in " << std::endl;
    for(int i=nWaitSec; i>0; --i){
        std::cout << i << std::endl;
        cv::waitKey(1000);
    }

    // std::ifstream posesFile("poses_office_2.txt");
    std::ifstream posesFile("poses_voest_short.txt");
    if(posesFile.is_open()) {
        std::cout << "File opened" << std::endl;
        std::string line;
        std::vector<std::string> tokens;

        while (std::getline(posesFile, line))
        {
            if (line.length() < 1)
                continue;
            tokens = split(line);
            // Transformation
            cv::Mat pose;
            cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);
            for(unsigned i = 0; i < 3; i++)
                for(unsigned j = 0; j < 4; j++)
                {
                    trans(i, j) =  toDouble(tokens[2 + i*4 + j].c_str()); // increm tokens i by 2 for office file
                }

            // cloudMapper.AddOrUpdateFrustum("frustum", trans, 0.5, 1, 0, 0, 2);
            cloudMapper.AddOrUpdateFrustum("frustum", trans, 1, 1, 1, 1, 2, true, 0.5, 0.5);
            cv::waitKey(75);

        }
    }

    return 0;
}

