#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

// g++ -std=c++1z 1_simple_facerec_eigenfaces.cpp -lopencv_face -lopencv_core -lopencv_imgcodecs -lstdc++fs

int main(int argc, char *argv[])
{
  std::vector<cv::Mat> images;
  std::vector<int>     labels;

  // Iterate through all subdirectories, looking for .pgm files
  fs::path p(argc > 1 ? argv[1] : "../att_faces");
  for (const auto &entry : fs::recursive_directory_iterator{ p }) {
    if (fs::is_regular_file(entry.status())) { // Was once always (wrongly) false in VS
      if (entry.path().extension() == ".pgm") {
        std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
        int label = atoi(str.c_str() + 1); // s1 -> 1 (pointer arithmetic)
        images.push_back(cv::imread(entry.path().string().c_str(), cv::IMREAD_GRAYSCALE));
        labels.push_back(label);
      }
    }
  }

  std::cout << " training...";
  cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->train(images, labels);  //

  //code used in opencv lab for live video
  cv::Mat frame;
  double fps = 30;
  const char win_name[] = "Who's there ? Face Recognition";

  cv::VideoCapture vid_in(0);   // argument is the camera id
  if (!vid_in.isOpened()) {
      std::cout << "error: Camera 0 could not be opened for capture.\n";
      return -1;
  }
  cv::namedWindow(win_name);

  cv::Rect cropRectangle(200, 110, 200, 250);
  cv::Mat resized;

  while (1) {
      vid_in >> frame;
      //program has been trained, live video is ON, now we have to put the frame at the good format 92*112 8bit pgm
      frame = frame(cropRectangle); //cropping the image to allow face recognition 
    
      imshow(win_name, frame); //displaying image

      cv::resize(frame, resized, cv::Size(92, 112)); //resizing after showing the frame to match
      cv::cvtColor(resized, resized, CV_BGR2GRAY); //changing to grayscale

      //both lines from opencv lab
      int predictedLabel = model->predict(resized); //predicting
      std::cout << "\nPredicted class = " << predictedLabel << '\n'; //which class has been predicted (should be 41 for me)

      if (cv::waitKeyEx(1000 / fps) >= 0) // how long to wait for a key (milliseconds)
          break;
  }

  vid_in.release(); //updating the video onthe screen;
  return 0;
}
