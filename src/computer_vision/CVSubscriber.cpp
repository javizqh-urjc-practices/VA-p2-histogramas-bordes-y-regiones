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

// Copied because it is only allowed to edit this file
// Compute the Discrete fourier transform
cv::Mat computeDFT(const cv::Mat &image)
{
  // Expand the image to an optimal sizeC
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
inline bool hist_in_screen = false;

inline std::string WINDOW_NAME = "Practica_5";
inline std::string WINDOW_HIST_NAME = "Histograms";

inline std::string MODE = "Option [0-4]";
inline std::string SHRINK_MIN = "Shrink min value [0-127]";
inline std::string SHRINK_MAX = "Shrink max value [128-255]";
inline std::string HOUGH = "Hough accumulator [0-255]";
inline std::string AREA = "Area [0-500]";

inline int CORRELATION = 0;

bool show_hsv_mode = true;
int GAUSSIAN = 3;

// One of the following block can be used as the hsv filter for blue lines
// int H_MIN = 86;
// int S_MIN = 19;
// int V_MIN = 113;
// int H_MAX = 107;
// int S_MAX = 39;
// int V_MAX = 250;

int H_MIN = 86;
int S_MIN = 10;
int V_MIN = 53;
int H_MAX = 107;
int S_MAX = 59;
int V_MAX = 250;

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

cv::Mat expandHistogram(const cv::Mat &image, const double max, const double min,
                        const int max_range, const int min_range)
{
cv::Mat tmp(image.rows, image.cols, CV_8U);

for (int i = 0; i < image.rows; i++) {
  for (int j = 0; j < image.cols; j++) {
    tmp.at<u_char>(i, j) = ((image.at<u_char>(i, j)  - min)/(max - min)) * (max_range - min_range) + min_range;
  }
}

return tmp;
}

cv::Mat contractHistogram(const cv::Mat &image, const double max, const double min,
                        const int shrink_max, const int shrink_min)
{
cv::Mat tmp(image.rows, image.cols, CV_8U);
for (int i = 0; i < image.rows; i++) { for (int j = 0; j < image.cols; j++) {
    tmp.at<u_char>(i, j) = ((shrink_max - shrink_min)/(max - min)) * (image.at<u_char>(i, j) - min) + shrink_min;
  }
}

return tmp;
}

void drawCentroid(std::vector<cv::Point> contour, cv::Scalar color, cv::Mat target_image)
{
  // Compute centroid
  cv::Moments m = cv::moments(contour);
  int cx = static_cast<int>(m.m10 / m.m00);
  int cy = static_cast<int>(m.m01 / m.m00);

  // Draw the mark into the image
  cv::circle(target_image, cv::Point(cx, cy), 5, color, -1);
  std::string text = std::to_string(contour.size());
  cv::putText(target_image, text, cv::Point(cx, cy - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

}

cv::Mat substractImages(const cv::Mat &image1, const cv::Mat &image2) {
// The images must be of equal size and type
cv::Mat tmp(image1.rows, image1.cols, CV_8U);

for (int i = 0; i < image1.rows; i++) {
  for (int j = 0; j < image1.cols; j++) {
    tmp.at<u_char>(i, j) = image1.at<u_char>(i, j) - image2.at<u_char>(i, j);
  }
}

return tmp;
}

void drawHistogram(int histSize, const cv::Mat& original, const cv::Mat& shrink,
                  const cv::Mat& substract, const cv::Mat& expand, const cv::Mat& equalized) {
// Draw the histograms for B, G and R
int hist_w = 512, hist_h = 400;
int bin_w = cvRound( (double) hist_w / histSize);
int shrink_min = cv::getTrackbarPos(CVParams::SHRINK_MIN, CVParams::WINDOW_NAME);
int shrink_max = cv::getTrackbarPos(CVParams::SHRINK_MAX, CVParams::WINDOW_NAME) + 128;

cv::Mat histImage(hist_h + 100, hist_w, CV_8UC3, cv::Scalar(0, 0, 0) );

// normalize the histograms between 0 and histImage.rows - 100
cv::normalize(original, original, 0, histImage.rows - 100, cv::NORM_MINMAX, -1, cv::Mat() );
cv::normalize(shrink, shrink, 0, histImage.rows - 100, cv::NORM_MINMAX, -1, cv::Mat() );
cv::normalize(substract, substract, 0, histImage.rows - 100, cv::NORM_MINMAX, -1, cv::Mat() );
cv::normalize(expand, expand, 0, histImage.rows - 100, cv::NORM_MINMAX, -1, cv::Mat() );
cv::normalize(equalized, equalized, 0, histImage.rows - 100, cv::NORM_MINMAX, -1, cv::Mat() );

// Draw the intensity line for histograms
for (int i = 1; i < histSize; i++) {
  cv::line(
    histImage, cv::Point(bin_w * (i - 1), hist_h + 100 - cvRound(original.at<float>(i - 1)) ),
    cv::Point(bin_w * (i), hist_h + 100 - cvRound(original.at<float>(i)) ),
    cv::Scalar(255, 0, 0), 2, 8, 0);
  cv::line(
    histImage, cv::Point(bin_w * (i - 1), hist_h + 100 - cvRound(shrink.at<float>(i - 1)) ),
    cv::Point(bin_w * (i), hist_h + 100 - cvRound(shrink.at<float>(i)) ),
    cv::Scalar(0, 0, 255), 2, 8, 0);
  cv::line(
    histImage, cv::Point(bin_w * (i - 1), hist_h + 100 - cvRound(substract.at<float>(i - 1)) ),
    cv::Point(bin_w * (i), hist_h + 100 - cvRound(substract.at<float>(i)) ),
    cv::Scalar(255, 255, 0), 2, 8, 0);
  cv::line(
    histImage, cv::Point(bin_w * (i - 1), hist_h + 100 - cvRound(expand.at<float>(i - 1)) ),
    cv::Point(bin_w * (i), hist_h + 100 - cvRound(expand.at<float>(i)) ),
    cv::Scalar(0, 255, 255), 2, 8, 0);
  cv::line(
    histImage, cv::Point(bin_w * (i - 1), hist_h + 100 - cvRound(equalized.at<float>(i - 1)) ),
    cv::Point(bin_w * (i), hist_h + 100 - cvRound(equalized.at<float>(i)) ),
    cv::Scalar(0, 255, 0), 2, 8, 0);
}

// Compare histograms
double comp_shrink = cv::compareHist(original, shrink, CVParams::CORRELATION);
double comp_substract = cv::compareHist(original, substract, CVParams::CORRELATION);
double comp_expand = cv::compareHist(original, expand, CVParams::CORRELATION);
double comp_equalized = cv::compareHist(original, equalized, CVParams::CORRELATION);

// Write text
std::string shrink_text = "Shrink [" + std::to_string(shrink_min) + "," + std::to_string(shrink_max)+"]: " + std::to_string(comp_shrink);
std::string substract_text = "Substract: " + std::to_string(comp_substract);
std::string expand_text = "Stretch: " + std::to_string(comp_expand);
std::string equalized_text = "Eqhist: " + std::to_string(comp_equalized);

cv::putText(histImage, shrink_text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
cv::putText(histImage, substract_text, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0));
cv::putText(histImage, expand_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
cv::putText(histImage, equalized_text, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

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

// parameters for window testing
// cv::createTrackbar("h", CVParams::WINDOW_NAME, nullptr, 255, 0);
// cv::createTrackbar("s", CVParams::WINDOW_NAME, nullptr, 255, 0);
// cv::createTrackbar("v", CVParams::WINDOW_NAME, nullptr, 255, 0);

// parameters for blue lines testing
// cv::createTrackbar("h_start", CVParams::WINDOW_NAME, nullptr, 255, 0);
// cv::createTrackbar("s_start", CVParams::WINDOW_NAME, nullptr, 255, 0);
// cv::createTrackbar("v_start", CVParams::WINDOW_NAME, nullptr, 255, 0);
// cv::createTrackbar("h_end", CVParams::WINDOW_NAME, nullptr, 255, 0);
// cv::createTrackbar("s_end", CVParams::WINDOW_NAME, nullptr, 255, 0);
// cv::createTrackbar("v_end", CVParams::WINDOW_NAME, nullptr, 255, 0);
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
int mode_param, shrink_min, shrink_max, hough_accumulator, area, num_of_lines;

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
hough_accumulator = cv::getTrackbarPos(CVParams::HOUGH, CVParams::WINDOW_NAME);
area = cv::getTrackbarPos(CVParams::AREA, CVParams::WINDOW_NAME);

// Option 1
cv::Mat bw, filter, complex_image, filtered_image, shrink_bw, difference, expand_bw, eq_bw;
cv::Mat hist_bw, hist_shrink, hist_subs, hist_expand, eqhist;

// Option 4
cv::Mat gray, edges, gauss, drawing, hsv_image, cloned_image, window_filter, lines_filter;
std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;
std::string text;

// Default color values
int h_window = 180; // 180 or 120
int s_window = 255; // 255 or 155
int v_window = 35; // 35 or 50

cv::Scalar color;
cv::Scalar lower(CVParams::H_MIN,CVParams::S_MIN,CVParams::V_MIN);
cv::Scalar upper(CVParams::H_MAX,CVParams::S_MAX,CVParams::V_MAX);

// ---- Use for selecting hsv for window ----
// Modify initWindow() for the trackers
  // h_window = cv::getTrackbarPos("h", CVParams::WINDOW_NAME);
  // s_window = cv::getTrackbarPos("s", CVParams::WINDOW_NAME);
  // v_window = cv::getTrackbarPos("v", CVParams::WINDOW_NAME);

// ---- Use for selecting hsv for blue lines ----
// Modify initWindow() for the trackers
// Remove or comment also the upper/lower declarations
  // h_start = cv::getTrackbarPos("h_start", CVParams::WINDOW_NAME);
  // s_start = cv::getTrackbarPos("s_start", CVParams::WINDOW_NAME);
  // v_start = cv::getTrackbarPos("v_start", CVParams::WINDOW_NAME);
  // h_end = cv::getTrackbarPos("h_end", CVParams::WINDOW_NAME);
  // s_end = cv::getTrackbarPos("s_end", CVParams::WINDOW_NAME);
  // v_end = cv::getTrackbarPos("v_end", CVParams::WINDOW_NAME);
  // cv::Scalar lower(h_start,s_start,v_start);
  // cv::Scalar upper(h_start,s_start,v_start);

// Establish the number of bins
int histSize = 256;
float range[] = {0, 256};       //the upper boundary is exclusive
const float * histRange = {range};
bool uniform = true, accumulate = false;

cv::Mat m;
double min, max;

// ----------------------------------------------------------------

if (mode_param != 1 && CVParams::hist_in_screen) {
  CVParams::hist_in_screen = false;
  cv::destroyWindow(CVParams::WINDOW_HIST_NAME);
}

switch (mode_param)
{
case 1:
  cv::cvtColor(in_image_rgb, bw, cv::COLOR_BGR2GRAY);
  cv::minMaxLoc(bw, &min, &max);
  cv::calcHist(&bw, 1, 0, cv::Mat(), hist_bw, 1, &histSize, &histRange, uniform, accumulate);

  // Filtering Spectrum
  complex_image = fftShift(computeDFT(bw));
  filter = CVFunctions::createLowPassFilter(complex_image, 20);
  cv::mulSpectrums(complex_image, filter, complex_image, 0);
  complex_image = fftShift(complex_image);
  cv::idft(complex_image, filtered_image, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(filtered_image, filtered_image, 0, 1, cv::NORM_MINMAX);

  // Calculating histogram and shrink
  shrink_bw = CVFunctions::contractHistogram(bw, max, min, shrink_max + 128, shrink_min);
  cv::calcHist(&shrink_bw, 1, 0, cv::Mat(), hist_shrink, 1, &histSize, &histRange, uniform, accumulate);

  // Combine the 2 images
  difference = CVFunctions::substractImages(bw, shrink_bw);
  cv::calcHist(&difference, 1, 0, cv::Mat(), hist_subs, 1, &histSize, &histRange, uniform, accumulate);

  // Expand the histogram
  expand_bw = CVFunctions::expandHistogram(difference, max, min, range[1], range[0]);
  cv::calcHist(&expand_bw, 1, 0, cv::Mat(), hist_expand, 1, &histSize, &histRange, uniform, accumulate);

  // Equalize the histogram
  cv::equalizeHist(expand_bw, eq_bw);
  cv::calcHist(&eq_bw, 1, 0, cv::Mat(), eqhist, 1, &histSize, &histRange, uniform, accumulate);

  CVParams::hist_in_screen = true;
  CVFunctions::drawHistogram(histSize, hist_bw, hist_shrink, hist_subs, hist_expand, eqhist);
  cv::imshow(CVParams::WINDOW_NAME, eq_bw);
  break;
case 2:
{
  cv::Mat edges, hsv, thresh;
  std::vector<cv::Vec2f> lines;   // will hold the results of the detection (rho, theta)

  // Filter the color of the clocks
  cv::cvtColor(in_image_rgb, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(h_window, s_window, v_window), thresh);
  cv::medianBlur(thresh, thresh, 3);

  // Edge detection
  cv::Canny(thresh, edges, 50, 120, 3);

  // Standard Hough Line Transform
  cv::HoughLines(edges, lines, 1, CVParams::PI / 180, hough_accumulator, 0, 0);   // runs the actual detection

  // Draw the lines
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    cv::Point pt1, pt2;
    double a = std::cos(theta), b = std::sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * ( a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * ( a));
    cv::line(out_image_rgb, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
  }

  cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
  break;
}
case 3:
{
  cv::Mat edges, hsv, thresh;
  std::vector<cv::Vec3f> circles;   // will hold the results of the detection (rho, theta)

  // Filter the color of the clocks
  cv::cvtColor(in_image_rgb, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(15, 100, 0), cv::Scalar(100, 255, 255), thresh);

  // Edge detection
  cv::medianBlur(thresh, thresh, 25);

  cv::HoughCircles(
    thresh, circles, cv::HOUGH_GRADIENT, 2,
    thresh.rows / 16,             // change this value to detect circles with different distances to each other
    100, 30, 5, 50              // change the last two parameters (min_radius & max_radius) to detect larger circles
  );

  // Draw the circles
  for (size_t i = 0; i < circles.size(); i++) {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    // circle center
    cv::circle(out_image_rgb, center, 1, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    // circle outline
    int radius = c[2];
    cv::circle(out_image_rgb, center, radius, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
  }

  cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
  break;
}
case 4:

  // Convert to HSV
  cv::cvtColor(in_image_rgb, hsv_image, cv::COLOR_BGR2HSV);

  // Filtering by color
  cv::inRange(hsv_image, lower, upper, lines_filter);
  cv::inRange(hsv_image, cv::Scalar(0, 0, 0), cv::Scalar(h_window, s_window, v_window), window_filter);
  filtered_image = lines_filter + window_filter;

  // Gaussian blur
  cv::GaussianBlur(filtered_image, gauss, cv::Size(CVParams::GAUSSIAN, CVParams::GAUSSIAN), 0);

  // Image processing
  cv::Canny(gauss, edges, 50, 100, 3);

  // Contours
  cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  // Drawing contours (only if > num of points)
  cloned_image = in_image_rgb.clone();
  num_of_lines = 0;

  for (size_t i = 0; i < contours.size(); i++) {
    if (static_cast<int>(contours[i].size()) > area)
    {
      num_of_lines++;

      color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
      cv::drawContours(cloned_image, contours, static_cast<int>(i), color, 4, cv::LINE_8);
      CVFunctions::drawCentroid(contours[i], color, cloned_image);
    }
  }

  // Writing header
  text = "Contours: " + std::to_string(num_of_lines);
  cv::putText(cloned_image, text, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

  cv::imshow(CVParams::WINDOW_NAME, cloned_image);
  break;

default:
  cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
  break;
}

cv::waitKey(3);

return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision
