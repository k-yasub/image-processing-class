#pragma once
#include <stdint.h>

enum YCCtype { YUV444, YUV420 };

const uint8_t HV[2][2] = {{0x11, 0x11}, {0x22, 0x11}};

enum Ctype { Luma, Chroma };