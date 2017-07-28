#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ void bilinear_interpolate(const Dtype* bottom_data,
    const int height, const int width, Dtype h, Dtype w,
    Dtype& maxval, Dtype& maxidx_h, Dtype& maxidx_w) {
  // Deal with cases that inverse elements are out of feature map boundary
  if (h < -0.5 || h > height - 0.5 || w < -0.5 || w > width - 0.5) {
    // empty
    return;
  }
  if (h <= 0) {
    h = Dtype(0.);
  }
  if (w <= 0) {
    w = Dtype(0.);
  }
  int h_low = static_cast<int>(h);
  int w_low = static_cast<int>(w);
  int h_high;
  int w_high;

  if (h_low >= height - 1) {
    h_high = height - 1;
    h_low = height - 1;
    h = Dtype(h_low);
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = width - 1;
    w_low = width - 1;
    w = Dtype(w_low);
  } else {
    w_high = w_low + 1;
  }

  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = Dtype(1.) - lh;
  Dtype hw = Dtype(1.) - lw;

  Dtype w1 = hh * hw;
  Dtype w2 = hh * lw;
  Dtype w3 = lh * hw;
  Dtype w4 = lh * lw;

  // Extract four values
  Dtype v1 = bottom_data[h_low * width + w_low];
  Dtype v2 = bottom_data[h_low * width + w_high];
  Dtype v3 = bottom_data[h_high * width + w_low];
  Dtype v4 = bottom_data[h_high * width + w_high];

  // Do bilinear interpolation
  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  if (val > maxval) {
    maxval = val;
    maxidx_h = h;
    maxidx_w = w;
  }
}

template <typename Dtype>
__global__ void ROIAlignPoolForward(
    const int nthreads, const Dtype* bottom_data,
    Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, Dtype* argmax_data,
    const int shape, const int sample_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * shape;
    Dtype x1 = bottom_rois[1];
    Dtype y1 = bottom_rois[2];
    Dtype x2 = bottom_rois[3] + Dtype(1.);
    Dtype y2 = bottom_rois[4] + Dtype(1.);

    // rescaled roi/mask size on conv feature map
    Dtype roi_start_w = round(x1 * spatial_scale);
    Dtype roi_start_h = round(y1 * spatial_scale);
    Dtype roi_end_w = round(x2 * spatial_scale);
    Dtype roi_end_h = round(y2 * spatial_scale);

    // Force malformed ROIs to be 1 x 1
    Dtype roi_width = max(roi_end_w - roi_start_w , Dtype(1.));
    Dtype roi_height = max(roi_end_h - roi_start_h , Dtype(1.));
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<int>(
        floor(static_cast<Dtype>(ph) * bin_size_h));
    Dtype wstart = static_cast<int>(
        floor(static_cast<Dtype>(pw) * bin_size_w));
    Dtype hend = static_cast<int>(
        ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
    Dtype wend = static_cast<int>(
        ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart+roi_start_h, Dtype(0.)), Dtype(height-1));
    hend = min(max(hend+roi_start_h, Dtype(0.)), Dtype(height-1));
    wstart = min(max(wstart+roi_start_w, Dtype(0.)), Dtype(width-1));
    wend = min(max(wend+roi_start_w, Dtype(0.)), Dtype(width-1));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    Dtype argmax_h = -FLT_MAX / 2;
    Dtype argmax_w = -FLT_MAX / 2;

    Dtype sample_h = bin_size_h / (sample_num + 1);
    Dtype sample_w = bin_size_w / (sample_num + 1);
    int roi_batch_ind = bottom_rois[0];
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    bool updated = false;
    for (int i = 1; i <= sample_num; ++i) {
      for (int j = 1; j <= sample_num; ++j) {
        Dtype cur_h = hstart + i * sample_h;
        Dtype cur_w = wstart + j * sample_w;
        if (cur_h >= hend || cur_w >= wend) {
          continue;
        }
        bilinear_interpolate(bottom_data,
            height, width, cur_h, cur_w,
            maxval, argmax_h, argmax_w);
        updated = true;
      }
    }
    top_data[index] = updated ? maxval : Dtype(0.);
    argmax_data[index*2+0] = argmax_h;
    argmax_data[index*2+1] = argmax_w;
  }
}

template <typename Dtype>
void ROIAlignPoolingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  int shape = bottom[1]->shape(1);
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* argmax_data = argmax_pos_.mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignPoolForward<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_, pooled_width_, bottom_rois,
      top_data, argmax_data, shape, sample_num_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignPoolBackward(
    const int nthreads, const Dtype* top_diff,
    Dtype* argmax_data, const int num_rois,
    const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois, const int shape) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = Dtype(0.);
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * shape;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }
      Dtype x1 = offset_bottom_rois[1];
      Dtype y1 = offset_bottom_rois[2];
      Dtype x2 = offset_bottom_rois[3]+1;
      Dtype y2 = offset_bottom_rois[4]+1;

      Dtype roi_start_w = round(x1 * spatial_scale);
      Dtype roi_start_h = round(y1 * spatial_scale);
      Dtype roi_end_w = round(x2 * spatial_scale);
      Dtype roi_end_h = round(y2 * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      bool in_roi = (
          w >= roi_start_w &&
          w <= roi_end_w &&
          h >= roi_start_h &&
          h <= roi_end_h);

      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data = argmax_data + offset * 2;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1 x 1
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          Dtype argmax_h = offset_argmax_data[(ph*pooled_width + pw)*2 + 0];
          Dtype argmax_w = offset_argmax_data[(ph*pooled_width + pw)*2 + 1];
          Dtype d_h = abs(argmax_h - h);
          Dtype d_w = abs(argmax_w - w);
          if (d_h < 1 && d_w < 1) {
            gradient += (1 - d_h) * (1 - d_w) *
                offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignPoolingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  int shape = bottom[1]->shape(1);
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  Dtype* argmax_data = argmax_pos_.mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignPoolBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_,
      channels_, height_, width_,
      pooled_height_, pooled_width_,
      bottom_diff, bottom_rois, shape);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignPoolingLayer);

}  // namespace caffe