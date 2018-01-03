#include <vector>

#include "caffe/layers/arrange_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ArrangeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
void ArrangeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  region_height_ = static_cast<int>(ceil(static_cast<float>(height_) / stride_));
  region_width_ = static_cast<int>(ceil(static_cast<float>(width_) / stride_));
  // top shape
  vector<int> newshape;
  newshape.push_back(num_);
  switch (arrangement_) {
    case ArrangeParameter_ArrangementMode_UNION:
      newshape.push_back(channels_);
      newshape.push_back(region_height_ * stride_);
      newshape.push_back(region_width_ * stride_);
      break;
    case ArrangeParameter_ArrangementMode_SPLIT:
      newshape.push_back(channels_ * stride_ * stride_);
      newshape.push_back(region_height_);
      newshape.push_back(region_width_);
      break;
    default:
      LOG(FATAL) << "Unknown arrangement mode: "
          << ArrangeParameter_ArrangementMode_Name(arrangement_);
  }
  
  top[0]->Reshape(newshape);
}

template <typename Dtype>
void ArrangeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // caffe_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
  caffe_set(top[0]->count(), Dtype(0), top_data);

  const int bottom_dim = bottom[0]->count(1);
  const int bottom_spatial_dim = bottom[0]->count(2);
  const int top_dim = top[0]->count(1);
  const int top_spatial_dim = top[0]->count(2);
  /***
  Memory Copy
  ***/
  switch (arrangement_) {
    case ArrangeParameter_ArrangementMode_UNION:
      for (int n = 0; n < num_; ++n) {
        int bottom_num_offset = n * bottom_dim;
        int top_num_offset = n * top_dim;

        for (int c = 0; c < channels_; ++c) {
          int bottom_channel_offset = c * bottom_spatial_dim;
          int top_channel_offset = c * top_spatial_dim;

          for (int h = 0; h < height_; ++h) {
            int region_h = h / stride_;
            int region_index_h = h % stride_;
            region_h += region_index_h * region_height_;
            // int region_h = h / stride_ + (h % stride_) * region_height_;
            int bottom_height_offset = h * width_;
            int top_height_offset = region_h * top[0]->width();

            for (int w = 0; w < width_; ++w) {
              int region_w = w / stride_;
              int region_index_w = w % stride_;
              region_w += region_index_w * region_width_;
              // int region_w = w / stride_ + (w % stride_) * region_width_;
              /***
              index = ((n * C + c) * H + h) * W + w
                    = n * C * H * W + c * H * W + h * W + w;
              ***/
              // int bottom_index = n * bottom_dim + c * bottom_spatial_dim + h * width_ + w;
              // int top_index = n * top_dim + c * top_spatial_dim + region_h * top[0]->width() + region_w;
              int bottom_index = bottom_num_offset + bottom_channel_offset + bottom_height_offset + w;
              int top_index = top_num_offset + top_channel_offset + top_height_offset + region_w;

              top_data[top_index] = bottom_data[bottom_index]
            }
          }
        }
      }
      break;
    case ArrangeParameter_ArrangementMode_SPLIT:
      const int region_dist = channels_ * top_spatial_dim;

      for (int n = 0; n < num_; ++n) {
        int bottom_num_offset = n * bottom_dim;
        int top_num_offset = n * top_dim;

        for (int c = 0; c < channels_; ++c) {
          int bottom_channel_offset = c * bottom_spatial_dim;
          int top_channel_offset = c * top_spatial_dim;

          for (int h = 0; h < height_; ++h) {
            int region_h = h / stride_;
            int region_index_h = h % stride_;

            int bottom_height_offset = h * width_;
            // int top_height_offset = region_h * top[0]->width();
            int top_height_offset = region_h * region_width_;

            for (int w = 0; w < width_; ++w) {
              int region_w = w / stride_;
              int region_index_w = w % stride_;

              int region_index = region_index_h * stride_ + region_index_w;
              int region_offset = region_index * region_dist;
              /***
              index = ((n * C + c) * H + h) * W + w
                    = n * C * H * W + c * H * W + h * W + w;
              ***/
              // int bottom_index = n * bottom_dim + c * bottom_spatial_dim + h * width_ + w;
              // int top_index = n * top_dim + region_index * region_dist + c * top_spatial_dim + region_h * region_width_ + region_w;
              int bottom_index = bottom_num_offset + bottom_channel_offset + bottom_height_offset + w;
              int top_index = top_num_offset + top_channel_offset + top_height_offset + region_w + region_offset;

              top_data[top_index] = bottom_data[bottom_index]
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown arrangement mode: "
          << ArrangeParameter_ArrangementMode_Name(arrangement_);
  }
}

template <typename Dtype>
void ArrangeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  const int bottom_dim = bottom[0]->count(1);
  const int bottom_spatial_dim = bottom[0]->count(2);
  const int top_dim = top[0]->count(1);
  const int top_spatial_dim = top[0]->count(2);
  /***
  Memory Copy
  ***/
  switch (arrangement_) {
    case ArrangeParameter_ArrangementMode_UNION:
      for (int n = 0; n < num_; ++n) {
        int bottom_num_offset = n * bottom_dim;
        int top_num_offset = n * top_dim;

        for (int c = 0; c < channels_; ++c) {
          int bottom_channel_offset = c * bottom_spatial_dim;
          int top_channel_offset = c * top_spatial_dim;

          for (int h = 0; h < height_; ++h) {
            int region_h = h / stride_;
            int region_index_h = h % stride_;
            region_h += region_index_h * region_height_;
            // int region_h = h / stride_ + (h % stride_) * region_height_;
            int bottom_height_offset = h * width_;
            int top_height_offset = region_h * top[0]->width();

            for (int w = 0; w < width_; ++w) {
              int region_w = w / stride_;
              int region_index_w = w % stride_;
              region_w += region_index_w * region_width_;
              // int region_w = w / stride_ + (w % stride_) * region_width_;
              /***
              index = ((n * C + c) * H + h) * W + w
                    = n * C * H * W + c * H * W + h * W + w;
              ***/
              // int bottom_index = n * bottom_dim + c * bottom_spatial_dim + h * width_ + w;
              // int top_index = n * top_dim + c * top_spatial_dim + region_h * top[0]->width() + region_w;
              int bottom_index = bottom_num_offset + bottom_channel_offset + bottom_height_offset + w;
              int top_index = top_num_offset + top_channel_offset + top_height_offset + region_w;

              bottom_diff[bottom_index] = top_diff[top_index]
            }
          }
        }
      }
      break;
    case ArrangeParameter_ArrangementMode_SPLIT:
      const int region_dist = channels_ * top_spatial_dim;

      for (int n = 0; n < num_; ++n) {
        int bottom_num_offset = n * bottom_dim;
        int top_num_offset = n * top_dim;

        for (int c = 0; c < channels_; ++c) {
          int bottom_channel_offset = c * bottom_spatial_dim;
          int top_channel_offset = c * top_spatial_dim;

          for (int h = 0; h < height_; ++h) {
            int region_h = h / stride_;
            int region_index_h = h % stride_;

            int bottom_height_offset = h * width_;
            // int top_height_offset = region_h * top[0]->width();
            int top_height_offset = region_h * region_width_;

            for (int w = 0; w < width_; ++w) {
              int region_w = w / stride_;
              int region_index_w = w % stride_;

              int region_index = region_index_h * stride_ + region_index_w;
              int region_offset = region_index * region_dist;
              /***
              index = ((n * C + c) * H + h) * W + w
                    = n * C * H * W + c * H * W + h * W + w;
              ***/
              // int bottom_index = n * bottom_dim + c * bottom_spatial_dim + h * width_ + w;
              // int top_index = n * top_dim + region_index * region_dist + c * top_spatial_dim + region_h * region_width_ + region_w;
              int bottom_index = bottom_num_offset + bottom_channel_offset + bottom_height_offset + w;
              int top_index = top_num_offset + top_channel_offset + top_height_offset + region_w + region_offset;

              bottom_diff[bottom_index] = top_diff[top_index]
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown arrangement mode: "
          << ArrangeParameter_ArrangementMode_Name(arrangement_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ArrangeLayer);
#endif

INSTANTIATE_CLASS(ArrangeLayer);
REGISTER_LAYER_CLASS(Arrange);

}  // namespace caffe
