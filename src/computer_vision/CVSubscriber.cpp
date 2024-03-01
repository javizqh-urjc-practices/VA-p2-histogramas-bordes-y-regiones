/*
 Copyright (c) 2023 José Miguel Guerrero Hernández

 Licensed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License;
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     https://creativecommons.org/licenses/by-sa/4.0/

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "computer_vision/CVSubscriber.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "rclcpp/rclcpp.hpp"

// -------- DFT Imported Functions ----------------------------------------------
// Copied because it is only allowed to edit this file

// Compute the Discrete fourier transform
cv::Mat computeDFT(const cv::Mat &image)
{
  // Expand the image to an optimal size.
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(image.rows);
  int n = cv::getOptimalDFTSize(image.cols);     // on the border add zero values
  cv::copyMakeBorder(
    image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(
      0));

  // Make place for both the complex and the real values
  cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);           // Add to the expanded another plane with zeros

  // Make the Discrete Fourier Transform
  cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);        // this way the result may fit in the source matrix
  return complexI;
}

// 6. Crop and rearrange
cv::Mat fftShift(const cv::Mat & magI)
{
  cv::Mat magI_copy = magI.clone();
  // crop the spectrum, if it has an odd number of rows or columns
  magI_copy = magI_copy(cv::Rect(0, 0, magI_copy.cols & -2, magI_copy.rows & -2));

  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = magI_copy.cols / 2;
  int cy = magI_copy.rows / 2;

  cv::Mat q0(magI_copy, cv::Rect(0, 0, cx, cy));     // Top-Left - Create a ROI per quadrant
  cv::Mat q1(magI_copy, cv::Rect(cx, 0, cx, cy));    // Top-Right
  cv::Mat q2(magI_copy, cv::Rect(0, cy, cx, cy));    // Bottom-Left
  cv::Mat q3(magI_copy, cv::Rect(cx, cy, cx, cy));   // Bottom-Right

  cv::Mat tmp;                             // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);                      // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  return magI_copy;
}


// Calculate dft spectrum
cv::Mat spectrum(const cv::Mat & complexI)
{
  cv::Mat complexImg = complexI.clone();
  // Shift quadrants
  cv::Mat shift_complex = fftShift(complexImg);

  // Transform the real and complex values to magnitude
  // compute the magnitude and switch to logarithmic scale
  // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  cv::Mat planes_spectrum[2];
  cv::split(shift_complex, planes_spectrum);         // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  cv::magnitude(planes_spectrum[0], planes_spectrum[1], planes_spectrum[0]);  // planes[0] = magnitude
  cv::Mat spectrum = planes_spectrum[0];

  // Switch to a logarithmic scale
  spectrum += cv::Scalar::all(1);
  cv::log(spectrum, spectrum);

  // Normalize
  cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX);   // Transform the matrix with float values into a
                                                      // viewable image form (float between values 0 and 1).
  return spectrum;
}

// -------- Parameters ----------------------------------------------

namespace CVParams {

  inline bool running = false;

  inline std::string WINDOW_NAME = "Practica_5";
  inline std::string WINDOW_HIST_NAME = "Histograms";

  inline std::string MODE = "Option [0-4]";
  inline std::string SHRINK_MIN = "Shrink min value [0-127]";
  inline std::string SHRINK_MAX = "Shrink max value [128-255]";
  inline std::string HOUGH = "Hough accumulator [0-255]";
  inline std::string AREA = "Area [0-500]";

  float PI = 3.14159265;
}

// -------- Self-Made Functions ----------------------------------------------
namespace CVFunctions {

cv::Mat createLowPassFilter(const cv::Mat &image, const int size)
{
  cv::Mat tmp(image.rows, image.cols, CV_32F);
  cv::Point center(image.rows / 2, image.cols / 2); // Is always even

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      if (size*size >= (center.x - i) * (center.x - i) + (center.y - j) * (center.y - j)) {
        tmp.at<float>(i, j) = 1;
      } else {
        tmp.at<float>(i, j) = 0;
      }
    }
  }

  cv::Mat toMerge[] = {tmp, tmp};
  cv::Mat horiz_Filter;
  cv::merge(toMerge, 2, horiz_Filter);

  return horiz_Filter;
}

cv::Mat contractHistogram(const cv::Mat &image, const double max, const double min,
                          const int shrink_max, const int shrink_min)
{
  cv::Mat tmp(image.rows, image.cols, CV_8U);

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      tmp.at<u_char>(i, j) = ((shrink_max - shrink_min)/(max - min)) * (image.at<u_char>(i, j) - min) + shrink_min;
    }
  }

  return tmp;
}

void drawHistogram(int histSize, const cv::Mat& original, const cv::Mat& shrink) {
  // Draw the histograms for B, G and R
  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound( (double) hist_w / histSize);

  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0) );

  // normalize the histograms between 0 and histImage.rows
  cv::normalize(original, original, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  cv::normalize(shrink, shrink, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

  // Draw the intensity line for histograms
  for (int i = 1; i < histSize; i++) {
    cv::line(
      histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(original.at<float>(i - 1)) ),
      cv::Point(bin_w * (i), hist_h - cvRound(original.at<float>(i)) ),
      cv::Scalar(255, 0, 0), 2, 8, 0);
    cv::line(
      histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(shrink.at<float>(i - 1)) ),
      cv::Point(bin_w * (i), hist_h - cvRound(shrink.at<float>(i)) ),
      cv::Scalar(0, 0, 255), 2, 8, 0);
  }

  // Show images
  cv::imshow(CVParams::WINDOW_HIST_NAME, histImage);
}

// -------- Window Management Functions ----------------------------------------------
void initWindow()
// Create window at the beggining
{
  if (CVParams::running) return;
  CVParams::running = true;

  // Show images in a different windows
  cv::namedWindow(CVParams::WINDOW_NAME);
  // create Trackbar and add to a window
  cv::createTrackbar(CVParams::MODE, CVParams::WINDOW_NAME, nullptr, 4, 0); 
  cv::createTrackbar(CVParams::SHRINK_MIN, CVParams::WINDOW_NAME, nullptr, 127, 0);
  cv::createTrackbar(CVParams::SHRINK_MAX, CVParams::WINDOW_NAME, nullptr, 127, 0);
  cv::createTrackbar(CVParams::HOUGH, CVParams::WINDOW_NAME, nullptr, 255, 0);
  cv::createTrackbar(CVParams::AREA, CVParams::WINDOW_NAME, nullptr, 500, 0);
}

}

namespace computer_vision
{

/**
   TO-DO: Default - the output images and pointcloud are the same as the input
 */
CVGroup CVSubscriber::processing(
  const cv::Mat in_image_rgb,
  const cv::Mat in_image_depth,
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
const
{
  int mode_param, shrink_min, shrink_max;

  // Create output images
  cv::Mat out_image_rgb, out_image_depth;
  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Processing
  out_image_rgb = in_image_rgb;
  out_image_depth = in_image_depth;
  out_pointcloud = in_pointcloud;

  // First time execution
  CVFunctions::initWindow();

  // Obtaining Parameter
  mode_param = cv::getTrackbarPos(CVParams::MODE, CVParams::WINDOW_NAME);
  shrink_min = cv::getTrackbarPos(CVParams::SHRINK_MIN, CVParams::WINDOW_NAME);
  shrink_max = cv::getTrackbarPos(CVParams::SHRINK_MAX, CVParams::WINDOW_NAME);

  // Option 1
  cv::Mat bw, filter, complex_image, filtered_image, hist_bw, hist_shrink, shrink_bw;
  // Establish the number of bins
  int histSize = 256;
  float range[] = {0, 256};       //the upper boundary is exclusive
  const float * histRange = {range};
  bool uniform = true, accumulate = false;

  cv::Mat m;
  double min, max;

  // ----------------------------------------------------------------

  switch (mode_param)
  {
  case 1:
    cv::cvtColor(in_image_rgb, bw, cv::COLOR_BGR2GRAY);

    // Filtering Spectrum 
    complex_image = fftShift(computeDFT(bw));
    filter = CVFunctions::createLowPassFilter(complex_image, 50);
    cv::mulSpectrums(complex_image, filter, complex_image, 0);
    complex_image = fftShift(complex_image);
    cv::idft(complex_image, filtered_image, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(filtered_image, filtered_image, 0, 1, cv::NORM_MINMAX);

    // Calculating histogram
    calcHist(&bw, 1, 0, cv::Mat(), hist_bw, 1, &histSize, &histRange, uniform, accumulate);
    cv::minMaxLoc(bw, &min, &max);
    shrink_bw = CVFunctions::contractHistogram(bw, max, min, shrink_max + 128, shrink_min);
    // cv::normalize(shrink_bw, shrink_bw, 0, 255, cv::NORM_MINMAX);
    calcHist(&shrink_bw, 1, 0, cv::Mat(), hist_shrink, 1, &histSize, &histRange, uniform, accumulate);


    CVFunctions::drawHistogram(histSize, hist_bw, hist_shrink);
    cv::imshow(CVParams::WINDOW_NAME, filtered_image);
    break;
  case 2:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  case 3:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  case 4:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  default:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  }

  cv::waitKey(3);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision
