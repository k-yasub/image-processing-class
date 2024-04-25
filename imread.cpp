#include <cstdio>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("An input image file is missing.\n");
    return EXIT_FAILURE;
  }
  cv::Mat image;
  image = cv::imread(argv[1], cv::ImreadModes::IMREAD_COLOR);
  if (image.empty()) {
    printf("Image file is not found.\n");
    return EXIT_FAILURE;
  }
  const int WIDTH = image.cols;
  const int HEIGHT = image.rows;
  const int NC = image.channels();
  printf("width = %d, height = %d", WIDTH, HEIGHT);
  for (int y = 0; y < HEIGHT;
       ++y) {  // ここのfor分をコメントアウトすると色味が元に戻る
    const int STRIDE = WIDTH * NC;
    for (int x = 0; x < WIDTH; ++x) {
      double out[3] = {0.0, 0.0, 0.0};
      double wY[3] = {0.114, 0.587, 0.299};
      double wCb[3] = {0.5, -.3313, -.1687};
      double wCr[3] = {-.0813, -.4187, 0.5};
      for (int c = 0; c < NC; ++c) {
        out[0] += image.data[y * STRIDE + x * NC + c] * wY[c];
        out[1] += image.data[y * STRIDE + x * NC + c] * wCb[c];
        out[2] += image.data[y * STRIDE + x * NC + c] * wCr[c];
      }
      for (int c = 0; c < NC; ++c) {
        int val = (c != 0) ? 128 : 0;
        image.data[y * STRIDE + x * NC + c] = out[c] + val;
      }
    }
  }
  cv::imshow("image", image);
  cv::waitKey();
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
