#include <vector>

#include "caffe/layers/arrange_back_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ArrangeBackLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ArrangeParameter& arrange_param = this->layer_param_.arrange_param();
  CHECK_GT(arrange_param.stride(), 1)
      << "stride must greater than 1.";
  stride_ = arrange_param.stride();
  if (!arrange_param.has_arrangement()) {
    arrangement_ = ArrangeParameter_ArrangementMode_UNION;
  } else {
    arrangement_ = arrange_param.arrangement();
  }
}

template <typename Dtype>
void ArrangeBackLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // top shape
  vector<int> newshape;
  newshape.push_back(num_);
  switch (arrangement_) {
    case ArrangeParameter_ArrangementMode_UNION:
      newshape.push_back(channels_);
      newshape.push_back(height_);
      newshape.push_back(width_);
      region_height_ = static_cast<int>(ceil(static_cast<float>(height_) / stride_));
      region_width_ = static_cast<int>(ceil(static_cast<float>(width_) / stride_));
      break;
    case ArrangeParameter_ArrangementMode_SPLIT:  
      CHECK_EQ(0, channels_ % (stride_ * stride_))
          << "Only support for channels_ % (stride_ * stride_) == 0 now";
      newshape.push_back(channels_ / stride_ / stride_);
      newshape.push_back(height_ * stride_);
      newshape.push_back(width_ * stride_);
      region_height_ = height_;
      region_width_ = width_;
      break;
    default:
      LOG(FATAL) << "Unknown arrangement mode: "
          << ArrangeParameter_ArrangementMode_Name(arrangement_);
  }
  top[0]->Reshape(newshape);
}

template <typename Dtype>
void ArrangeBackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // caffe_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
  caffe_set(top[0]->count(), Dtype(0), top_data);

  // const int bottom_dim = bottom[0]->count(1);
  const int bottom_spatial_dim = bottom[0]->count(2);
  // const int top_dim = top[0]->count(1);
  const int top_spatial_dim = top[0]->count(2);
  const int top_channels = top[0]->channels();
  const int top_width = top[0]->width();
  /***
  Memory Copy
  ***/
  switch (arrangement_) {
    case ArrangeParameter_ArrangementMode_UNION:
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int h = 0; h < height_; ++h) {
            int region_index_h = h / region_height_;
            int top_h = (h - region_index_h * region_height_) * stride_ + region_index_h;
            for (int w = 0; w < width_; ++w) {
              int region_index_w = w / region_width_;
              int top_w = (w - region_index_w * region_width_) * stride_ +  region_index_w;
              top_data[top_h * top_width + top_w] = bottom_data[h * width_ + w];
            }
          }
          // compute offset
          bottom_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
        } 
      }
      break;
    case ArrangeParameter_ArrangementMode_SPLIT:
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
          int region_index = c / top_channels;
          int region_index_h = region_index / stride_;
          int region_index_w = region_index % stride_;
          int top_channel_index = c % top_channels;
          int bottom_channel_offset = c * bottom_spatial_dim;
          int top_channel_offset = top_channel_index * top_spatial_dim;

          for (int h = 0; h < height_; ++h) {
            int top_h = h * stride_ + region_index_h;
            for (int w = 0; w < width_; ++w) {
              int top_w = w * stride_ + region_index_w;

              top_data[top_channel_offset + top_h * top_width + top_w] = bottom_data[bottom_channel_offset + h * width_ + w];
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(1);
        top_data += top[0]->offset(1);
      }
      break;
    default:
      LOG(FATAL) << "Unknown arrangement mode: "
          << ArrangeParameter_ArrangementMode_Name(arrangement_);
  }
}

template <typename Dtype>
void ArrangeBackLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // const int bottom_dim = bottom[0]->count(1);
  const int bottom_spatial_dim = bottom[0]->count(2);
  // const int top_dim = top[0]->count(1);
  const int top_spatial_dim = top[0]->count(2);
  const int top_channels = top[0]->channels();
  const int top_width = top[0]->width();
  /***
  Memory Copy
  ***/
  switch (arrangement_) {
    case ArrangeParameter_ArrangementMode_UNION:
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int h = 0; h < height_; ++h) {
            int region_index_h = h / region_height_;
            int top_h = (h - region_index_h * region_height_) * stride_ + region_index_h;
            for (int w = 0; w < width_; ++w) {
              int region_index_w = w / region_width_;
              int top_w = (w - region_index_w * region_width_) * stride_ +  region_index_w;
              bottom_diff[h * width_ + w] = top_diff[top_h * top_width + top_w];
            }
          }
          // compute offset
          bottom_diff += bottom[0]->offset(0, 1);
          top_diff += top[0]->offset(0, 1);
        } 
      }
      break;
    case ArrangeParameter_ArrangementMode_SPLIT:
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < channels_; ++c) {
          int region_index = c / top_channels;
          int region_index_h = region_index / stride_;
          int region_index_w = region_index % stride_;
          int top_channel_index = c % top_channels;
          int bottom_channel_offset = c * bottom_spatial_dim;
          int top_channel_offset = top_channel_index * top_spatial_dim;

          for (int h = 0; h < height_; ++h) {
            int top_h = h * stride_ + region_index_h;
            for (int w = 0; w < width_; ++w) {
              int top_w = w * stride_ + region_index_w;
              bottom_diff[bottom_channel_offset + h * width_ + w] = top_diff[top_channel_offset + top_h * top_width + top_w];
            }
          }
        }
        // compute offset
        bottom_diff += bottom[0]->offset(1);
        top_diff += top[0]->offset(1);
      }
      break;
    default:
      LOG(FATAL) << "Unknown arrangement mode: "
          << ArrangeParameter_ArrangementMode_Name(arrangement_);
  }
}

// #ifdef CPU_ONLY
// STUB_GPU(ArrangeBackLayer);
// #endif

INSTANTIATE_CLASS(ArrangeBackLayer);
REGISTER_LAYER_CLASS(ArrangeBack);

}  // namespace caffe
