#pragma once

#include <stdint.h>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <vector>

#include "bitstream.hpp"
#include "huffman_tables.hpp"
#include "ycctype.hpp"
#include "zigzag_order.hpp"

void encode_block(cv::Mat &, int, int &, bitstream &);

void encode_DC(int diff, const uint32_t *Ctable, const uint32_t *Ltable,
               bitstream &);
void encode_AC(int run, int val, const uint32_t *Ctable, const uint32_t *Ltable,
               bitstream &);

void Encode_MCU(std::vector<cv::Mat> &in, bitstream &enc, int YCC = YUV420) {
  constexpr int BS = 8;
  const int width = in[0].cols;
  const int height = in[0].rows;

  int Hblk = HV[YCC][0] & 0xF;  // 下位4bitは水平
  int Vblk = HV[YCC][0] >> 4;   // 上位4bitは垂直
  if (in.size() == 1) {
    Hblk = 1;
    Vblk = 1;
  }
  int prev_dc[3] = {0};
  for (int y = 0, cy = 0; y < height; y += BS * Vblk, cy += BS) {
    for (int x = 0, cx = 0; x < width; x += BS * Hblk, cx += BS) {
      cv::Mat blk;
      // Y
      for (int v = 0; v < Vblk; ++v) {
        for (int h = 0; h < Hblk; ++h) {
          blk = in[0](cv::Rect(x + BS * h, y + BS * v, BS, BS)).clone();
          encode_block(blk, Luma, prev_dc[0], enc);
        }
      }
      if (in.size() == 3) {
        // Cb
        blk = in[2](cv::Rect(cx, cy, BS, BS)).clone();
        encode_block(blk, Chroma, prev_dc[1], enc);
        // Cr
        blk = in[1](cv::Rect(cx, cy, BS, BS)).clone();
        encode_block(blk, Chroma, prev_dc[2], enc);
      }
    }
  }
}

void encode_block(cv::Mat &in, int c, int &prev_dc, bitstream &enc) {
  int16_t *p = (int16_t *)in.data;
  // DPCM
  int diff = p[0] - prev_dc;
  prev_dc = p[0];
  encode_DC(diff, DC_cwd[c], DC_len[c], enc);

  // loop for AC
  int run = 0;
  for (int i = 1; i < 64; i++) {
    int ac = p[scan[i]];
    if (ac == 0) {
      run++;
    } else {
      while (run > 15) {
        // ZRL
        encode_AC(0xF, 0x0, AC_cwd[c], AC_len[c], enc);
        run -= 16;
      }
      encode_AC(run, ac, AC_cwd[c], AC_len[c], enc);
      run = 0;
    }
  }
  if (run) {
    // EOB
    encode_AC(0x0, 0x0, AC_cwd[c], AC_len[c], enc);
  }
}

void encode_DC(int diff, const uint32_t *Ctable, const uint32_t *Ltable,
               bitstream &enc) {
  int s = 0;                             // 係数のビット長
  int uval = (diff < 0) ? -diff : diff;  // uval = diffの絶対値
  int bound = 1;
  while (uval >= bound) {
    bound += bound;
    s++;
  }
  enc.put_bits(Ctable[s], Ltable[s]);
  if (s != 0) {
    if (diff < 0) {
      diff -= 1;
    }
    enc.put_bits(diff, s);
  }
}

void encode_AC(int run, int val, const uint32_t *Ctable, const uint32_t *Ltable,
               bitstream &enc) {
  int s = 0;                          // 係数のビット長
  int uval = (val < 0) ? -val : val;  // uval = diffの絶対値
  int bound = 1;
  while (uval >= bound) {
    bound += bound;
    s++;
  }
  enc.put_bits(Ctable[(run << 4) + s], Ltable[(run << 4) + s]);
  if (s != 0) {
    if (val < 0) {
      val -= 1;
    }
    enc.put_bits(val, s);
  }
}
