/*
 * Copyright (c) 2019 Flight Dynamics and Control Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cstdlib>

#include "fdcl_common.hpp"

#include <librealsense2/rs.hpp>


int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv, fdcl::keys);

    const char* about = "Pose estimation of ArUco marker images";

    auto success = parse_inputs(parser, about);
    if (!success) {
        return 1;
    }

    // If -v is provided, use VideoCapture (offline). Otherwise use RealSense.
    cv::VideoCapture in_video;

    rs2::pipeline rs_pipe;
    rs2::config rs_cfg;
    bool useVideoFile = parser.has("v");   // assuming parse_video_in uses -v
    bool rs_started = false;

    if (useVideoFile) {
        success = parse_video_in(in_video, parser);
        if (!success) return 1;
    } else {
        rs_cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        try {
            rs_pipe.start(rs_cfg);
            rs_started = true;
        } catch (const rs2::error& e) {
            std::cerr << "RealSense error calling " << e.get_failed_function()
                    << "(" << e.get_failed_args() << "):\n"
                    << e.what() << std::endl;
            return 1;
        }
    }

    int dictionary_id = parser.get<int>("d");
    float marker_length_m = parser.get<float>("l");
    int wait_time = 10;

    if (marker_length_m <= 0) {
        std::cerr << "Marker length must be a positive value in meter\n";
        return 1;
    }

    cv::Mat image, image_copy;
    cv::Mat camera_matrix, dist_coeffs;

    std::ostringstream vector_to_marker;

    // Create the dictionary from the same dictionary the marker was generated.
    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary( \
        cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionary_id));


    cv::FileStorage fs("../../calibration_params.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;


    for (;;)
    {
        if (useVideoFile) {
            if (!in_video.grab()) break;
            in_video.retrieve(image);
            if (image.empty()) break;
        } else {
            rs2::frameset frames = rs_pipe.wait_for_frames();
            rs2::video_frame color = frames.get_color_frame();
            if (!color) continue;

            cv::Mat rs_image(
                cv::Size(color.get_width(), color.get_height()),
                CV_8UC3,
                (void*)color.get_data(),
                cv::Mat::AUTO_STEP
            );
            image = rs_image.clone(); // safe copy
        }

        image.copyTo(image_copy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);

        // if at least one marker detected
        if (ids.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(image_copy, corners, ids);

            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_m,
                    camera_matrix, dist_coeffs, rvecs, tvecs);
                    
            std::cout << "Translation: " << tvecs[0]
                << "\tRotation: " << rvecs[0] << "\n";
            
            // Draw axis for each marker
            for(int i=0; i < ids.size(); i++)
            {
                cv::aruco::drawAxis(image_copy, camera_matrix, dist_coeffs,
                        rvecs[i], tvecs[i], 0.1);

                // This section is going to print the data for the first the 
                // detected marker. If you have more than a single marker, it is 
                // recommended to change the below section so that either you
                // only print the data for a specific marker, or you print the
                // data for each marker separately.
                drawText(image_copy, "x", tvecs[0](0), cv::Point(10, 30));
                drawText(image_copy, "y", tvecs[0](1), cv::Point(10, 50));
                drawText(image_copy, "z", tvecs[0](2), cv::Point(10, 70));
            }
        }

        imshow("Pose estimation", image_copy);
        char key = (char)cv::waitKey(wait_time);
        if (key == 27) {
            break;
        }
    }

    if (useVideoFile) in_video.release();
    if (rs_started) rs_pipe.stop();

    return 0;
}
