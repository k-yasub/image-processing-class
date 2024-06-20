#pragma once
#include <functional>
#include <opencv2/core.hpp>

#define BSIZE 8
// ブロック処理のための関数: 実際の処理はfuncで決まる
void blkproc(cv::Mat &in, std::function<void(cv::Mat &, float *)> func,
             float *fp = nullptr) {
  for (int y = 0; y < in.rows; y += BSIZE) {
    for (int x = 0; x < in.cols; x += BSIZE) {
      cv::Mat blk_in = in(cv::Rect(x, y, BSIZE, BSIZE)).clone();
      cv::Mat blk_out = in(cv::Rect(x, y, BSIZE, BSIZE));
      func(blk_in, fp);
      blk_in.convertTo(blk_out, blk_out.type());
    }
  }
}

void fdct2(cv::Mat &in, float *fp = nullptr) { cv::dct(in, in); }
void idct2(cv::Mat &in, float *fp = nullptr) { cv::idct(in, in); }

void quantization(cv::Mat &in, float *qtable) {
  if (qtable == nullptr) {
    printf("qtable is missing.\n");
    exit(EXIT_FAILURE);
  }
  float *p = (float *)in.data;
  for (int y = 0; y < in.rows; ++y) {
    for (int x = 0; x < in.cols; ++x) {
      float val = p[y * in.rows + x];
      float a = fabs(val);
      a /= qtable[y * in.rows + x];
      a = floorf(a);
      a *= (val < 0.0) ? -1 : 1;
      p[y * in.rows + x] = a;
    }
  }
}

void dequantization(cv::Mat &in, float *qtable) {
  float *p = (float *)in.data;
  for (int y = 0; y < in.rows; ++y) {
    for (int x = 0; x < in.cols; ++x) {
      float val = p[y * in.rows + x];
      float a = fabs(val);
      if (a > 0.0) {
        a += 0.5;
      }
      a *= qtable[y * in.rows + x];
      a = floorf(a);
      a *= (val < 0.0) ? -1 : 1;
      p[y * in.rows + x] = a;
    }
  }
}

void iminfo(cv::Mat &in) {
  int width = in.cols;
  int height = in.rows;
  int num_components = in.channels();
  std::cout << "width = " << width << ", ";
  std::cout << "height = " << height << ", ";
  if (num_components == 3) {
    std::cout << "COLOR" << std::endl;
  } else if (num_components == 1) {
    std::cout << "GRAYSCALE" << std::endl;
  } else {
    std::cout << "UNKOWN COLORSPACE" << std::endl;
  }
}

template <class T>
T clip(T pixel) {
  if (pixel < 0) {
    pixel = 0;
  } else if (pixel > 255) {
    pixel = 255;
  }
  return pixel;
}
template <>
float clip(float pixel) {
  if (pixel < 0.0f) {
    pixel = 0.0f;
  } else if (pixel > 255.0f) {
    pixel = 255.0f;
  }
  return pixel;
}
template <>
double clip(double pixel) {
  if (pixel < 0.0) {
    pixel = 0.0;
  } else if (pixel > 255.0) {
    pixel = 255.0;
  }
  return pixel;
}
