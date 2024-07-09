#include <chrono>
#include <climits>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "entropy-coding.hpp"
#include "jpgheaders.hpp"
#include "mytools.hpp"
#include "qtable.hpp"

static long long time_dct = 0;
static long long time_qnt = 0;
static long long time_enc = 0;

#define TIC() std::chrono::high_resolution_clock::now()
#define TOC(x)                                           \
  std::chrono::duration_cast<std::chrono::microseconds>( \
      std::chrono::high_resolution_clock::now() - (x))   \
      .count()

void process_component_enc(cv::Mat &image, int *qmatrix) {
  cv::Mat fimage;
  image.convertTo(fimage, CV_32F);
  fimage -= 128.0;
  auto t0 = TIC();
  blkproc(fimage, fdct2);
  auto t1 = TOC(t0);
  time_dct += t1;

  auto t2 = TIC();
  blkproc(fimage, quantization, qmatrix);
  auto t3 = TOC(t2);
  time_qnt += t3;

  fimage.convertTo(image, CV_16S);
}

void process_component_dec(cv::Mat &image, int *qmatrix) {
  cv::Mat fimage;
  image.convertTo(fimage, CV_32F);
  blkproc(fimage, dequantization, qmatrix);
  blkproc(fimage, idct2);
  fimage += 128.0;
  fimage.convertTo(image, CV_8U);
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
  int qmatrix[3][64];
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 64; ++i) {
      qmatrix[c][i] = static_cast<int>(clip(round(qtable[c][i] * scale)));
      if (qmatrix[c][i] == 0) {
        qmatrix[c][i] = 1;
      }
    }
  }

  // メインヘッダの生成
  bitstream enc;
  create_mainheader(image.cols, image.rows, image.channels(), qmatrix[Luma],
                    qmatrix[Chroma], YUV420, enc);

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
    cv::resize(cimage[2], cimage[2], cv::Size(), 0.5, 0.5, cv::INTER_AREA);
  }

  for (int c = 0; c < image.channels(); ++c) {
    process_component_enc(cimage[c], qmatrix[c]);
  }

  // entropy coding
  auto t4 = TIC();
  Encode_MCU(cimage, enc, YUV420);
  auto t5 = TOC(t4);
  time_enc += t5;

  // .jpgの書き出し
  FILE *fp = fopen("output.jpg", "wb");
  if (fp == NULL) {
    // error
    printf("file open error\n");
    exit(EXIT_FAILURE);
  }
  auto codestream = enc.finalize();
  fwrite(codestream.data(), sizeof(uint8_t), codestream.size(), fp);
  fclose(fp);

  for (int c = 0; c < image.channels(); ++c) {
    process_component_dec(cimage[c], qmatrix[c]);
  }

  // クロマアップサンプリング
  if (image.channels() == 3) {
    cv::resize(cimage[1], cimage[1], cv::Size(), 2.0, 2.0, cv::INTER_AREA);
    cv::resize(cimage[2], cimage[2], cv::Size(), 2.0, 2.0, cv::INTER_AREA);
  }
  cv::merge(cimage, image);
  if (image.channels() == 3) {
    cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
  }
  printf("PSNR = %f [dB]\n", PSNR(original, image));
  printf("Codestream length = %d Bytes\n", codestream.size());
  printf("DCT: %7.3f [ms]\n", time_dct / 1000.0);
  printf("Quantization: %7.3f [ms]\n", time_qnt / 1000.0);
  printf("Entropy: %7.3f [ms]\n", time_enc / 1000.0);
  // cv::imshow("Output", image);
  // int keycode = 0;
  // while (keycode != 'q') {
  //   keycode = cv::waitKey(0);
  // }
}

// 横軸bpp,縦軸PSNRで結果を出す
// csvに起こしてpythonで作ればよさそう
// DCT、Quantization、Entropyの実行時間を出力
