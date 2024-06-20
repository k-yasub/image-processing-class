#include <climits>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mytools.hpp"
#include "qtable.hpp"

void process_component(cv::Mat &image, float *qmatrix) {
  cv::Mat fimage;
  image.convertTo(fimage, CV_32F);
  fimage -= 128.0;
  blkproc(fimage, fdct2);
  blkproc(fimage, quantization, qmatrix);
  blkproc(fimage, dequantization, qmatrix);
  blkproc(fimage, idct2);
  fimage += 128.0;
  fimage.convertTo(image, image.type());
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Input file is not specified." << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat image;
  image = cv::imread(argv[1], cv::ImreadModes::IMREAD_ANYCOLOR);
  if (image.empty()) {
    std::cerr << "Image file is not found." << std::endl;
    return EXIT_FAILURE;
  }

  int quality = 75;
  if (argc > 2) {
    quality = strtol(argv[2], nullptr, 10);
  }
  float QF;
  if (quality <= 50) {
    QF = floorf(5000.0f / quality);
  } else {
    QF = 200.0f - 2.0f * quality;
  }
  float scale = QF / 100.0f;
  float qmatrix[3][64];
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 64; ++i) {
      qmatrix[c][i] = clip(round(qtable[c][i] * scale));
      if (qmatrix[c][i] < FLT_EPSILON) {
        qmatrix[c][i] = 1.0;
      }
    }
  }
  iminfo(image);
  cv::Mat original = image.clone();

  if (image.channels() == 3) {
    cv::cvtColor(image, image, cv::COLOR_BGR2YCrCb);
  }
  std::vector<cv::Mat> cimage(image.channels());
  // カラー画像の場合、cimage[0] = R, cimage[1] = G...
  cv::split(image, cimage);
  // クロマサブサンプリング
  if (image.channels() == 3) {
    cv::resize(cimage[1], cimage[1], cv::Size(), 0.5, 0.5, cv::INTER_AREA);
  }
  for (int c = 0; c < image.channels(); ++c) {
    process_component(cimage[c], qmatrix[c]);
  }
  // クロマアップサンプリング
  if (image.channels() == 3) {
    cv::resize(cimage[1], cimage[1], cv::Size(), 2.0, 2.0, cv::INTER_AREA);
  }
  cv::merge(cimage, image);
  if (image.channels() == 3) {
    cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
  }
  printf("PSNR = %f [dB]\n", PSNR(original, image));
  cv::imshow("Output", image);
  int keycode = 0;
  while (keycode != 'q') {
    keycode = cv::waitKey(0);
  }
}
