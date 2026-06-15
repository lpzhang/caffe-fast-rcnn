#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/nms_heatmap_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void NMSHeatmapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NMSHeatmapParameter nms_heatmap_param = this->layer_param_.nms_heatmap_param();

  CHECK(!nms_heatmap_param.has_kernel_size() !=
      !(nms_heatmap_param.has_kernel_h() && nms_heatmap_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(nms_heatmap_param.has_kernel_size() ||
      (nms_heatmap_param.has_kernel_h() && nms_heatmap_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  if (nms_heatmap_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = nms_heatmap_param.kernel_size();
  } else {
    kernel_h_ = nms_heatmap_param.kernel_h();
    kernel_w_ = nms_heatmap_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  CHECK_EQ(kernel_h_ % 2, 1) << "Filter dimensions cannot be even number.";
  CHECK_EQ(kernel_w_ % 2, 1) << "Filter dimensions cannot be even number.";

  pad_h_ = kernel_h_ / 2;
  pad_w_ = kernel_w_ / 2;
  stride_h_ = 1;
  stride_w_ = 1;

  // beta_ as the threshold value.
  beta_ = nms_heatmap_param.has_beta() ? nms_heatmap_param.beta() : Dtype(0.0);
}

template <typename Dtype>
void NMSHeatmapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  CHECK_EQ(pooled_height_, height_) << "Pooled dimensions must be equal to the original dimensions.";
  CHECK_EQ(pooled_width_, width_) << "Pooled dimensions must be equal to the original dimensions.";

  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // Initialize
  if (top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
}


template <typename Dtype>
void NMSHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Initialize
  if (use_top_mask) {
    top_mask = top[1]->mutable_cpu_data();
    caffe_set(top_count, Dtype(-1), top_mask);
  } else {
    mask = max_idx_.mutable_cpu_data();
    caffe_set(top_count, -1, mask);
  }
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = min(hstart + kernel_h_, height_);
          int wend = min(wstart + kernel_w_, width_);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          // find out the local maximum
          Dtype maxval = Dtype(-FLT_MAX);
          int maxidx = -1;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              if (bottom_data[h * width_ + w] > maxval) {
                maxidx = h * width_ + w;
                maxval = bottom_data[maxidx];
              }
            }
          }
          // if the current position (pool_index) is the local maximum
          // and its value is not less than beta_
          const int pool_index = ph * pooled_width_ + pw;
          if (pool_index == maxidx && maxval >= beta_) {
            top_data[pool_index] = maxval;
            if (use_top_mask) {
              top_mask[pool_index] = static_cast<Dtype>(maxidx);
            } else {
              mask[pool_index] = maxidx;
            }
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      if (use_top_mask) {
        top_mask += top[0]->offset(0, 1);
      } else {
        mask += top[0]->offset(0, 1);
      }
    }
  }
}

template <typename Dtype>
void NMSHeatmapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  // The main loop
  if (use_top_mask) {
    top_mask = top[1]->cpu_data();
  } else {
    mask = max_idx_.cpu_data();
  }
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const int index = ph * pooled_width_ + pw;
          const int bottom_index = use_top_mask ? top_mask[index] : mask[index];
          if (bottom_index >= 0) {
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
      }
      // compute offset
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
      if (use_top_mask) {
        top_mask += top[0]->offset(0, 1);
      } else {
        mask += top[0]->offset(0, 1);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NMSHeatmapLayer);
#endif

INSTANTIATE_CLASS(NMSHeatmapLayer);

}  // namespace caffe
