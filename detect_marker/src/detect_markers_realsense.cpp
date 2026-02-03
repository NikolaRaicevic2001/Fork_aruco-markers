#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <librealsense2/rs.hpp>
#include <iostream>
#include <cstdlib>

#include "fdcl_common.hpp"

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv, fdcl::keys);

    const char* about = "Detect ArUco marker images";
    auto success = parse_inputs(parser, about);
    if (!success) return 1;

    int dictionary_id = parser.get<int>("d");
    int wait_time = 1;

    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary(
            cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionary_id));

    // RealSense pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);

    pipe.start(cfg);

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color = frames.get_color_frame();

        cv::Mat image(
            cv::Size(color.get_width(), color.get_height()),
            CV_8UC3,
            (void*)color.get_data(),
            cv::Mat::AUTO_STEP
        );

        cv::Mat image_copy = image.clone();

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        cv::aruco::detectMarkers(image_copy, dictionary, corners, ids);

        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(image_copy, corners, ids);
        }

        cv::imshow("Detected markers", image_copy);
        char key = (char)cv::waitKey(wait_time);
        if (key == 27) break;
    }

    pipe.stop();
    return 0;
}
