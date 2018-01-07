#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/arrange_back_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UnionModeBackForward(const int nthreads, const Dtype* const bottom_data,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width, const int stride,
    const int top_channels, const int top_height, const int top_width, const int top_dim, const int top_spatial_dim,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
   
    int region_index_h = h / region_height;
    int region_index_w = w / region_width;
    int top_h = (h - region_index_h * region_height) * stride + region_index_h;
    int top_w = (w - region_index_w * region_width) * stride +  region_index_w;

    Dtype* const top_slice = top_data + n * top_dim + c * top_spatial_dim;
    top_slice[top_h * top_width + top_w] = bottom_data[index];
  }
}

template <typename Dtype>
__global__ void SplitModeBackForward(const int nthreads, const Dtype* const bottom_data,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width, const int stride,
    const int top_channels, const int top_height, const int top_width, const int top_dim, const int top_spatial_dim,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    int region_index = c / top_channels;
    int region_index_h = region_index / stride;
    int region_index_w = region_index % stride;
    int top_channel_index = c % top_channels;

    int top_h = h * stride + region_index_h;
    int top_w = w * stride + region_index_w;

    Dtype* const top_slice = top_data + n * top_dim + top_channel_index * top_spatial_dim;
    top_slice[top_h * top_width + top_w] = bottom_data[index];
  }
}

template <typename Dtype>
void ArrangeBackLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);

  const int count = bottom[0]->count();
  const int top_dim = top[0]->count(1);
  const int top_spatial_dim = top[0]->count(2);
  const int top_channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
 
  switch (arrangement_) {
  case ArrangeParameter_ArrangementMode_UNION:
    // NOLINT_NEXT_LINE(whitespace/operators)
    UnionModeBackForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,
        bottom[0]->num(), channels_, height_, width_,
        region_height_, region_width_, stride_,
        top_channels, top_height, top_width, top_dim, top_spatial_dim,
        top_data);
    break;
  case ArrangeParameter_ArrangementMode_SPLIT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SplitModeBackForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,
        bottom[0]->num(), channels_, height_, width_,
        region_height_, region_width_, stride_,
        top_channels, top_height, top_width, top_dim, top_spatial_dim,
        top_data);
    break;
  default:
    LOG(FATAL) << "Unknown Arrange Mode.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void UnionModeBackBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width, const int stride,
    const int top_channels, const int top_height, const int top_width, const int top_dim, const int top_spatial_dim,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    int region_index_h = h / region_height;
    int region_index_w = w / region_width;
    int top_h = (h - region_index_h * region_height) * stride + region_index_h;
    int top_w = (w - region_index_w * region_width) * stride +  region_index_w;

    const Dtype* const top_slice = top_diff + n * top_dim + c * top_spatial_dim;
    bottom_diff[index] = top_slice[top_h * top_width + top_w];
  }
}

template <typename Dtype>
__global__ void SplitModeBackBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width, const int stride,
    const int top_channels, const int top_height, const int top_width, const int top_dim, const int top_spatial_dim,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    int region_index = c / top_channels;
    int region_index_h = region_index / stride;
    int region_index_w = region_index % stride;
    int top_channel_index = c % top_channels;

    int top_h = h * stride + region_index_h;
    int top_w = w * stride + region_index_w;

    const Dtype* const top_slice = top_diff + n * top_dim + top_channel_index * top_spatial_dim;
    bottom_diff[index] = top_slice[top_h * top_width + top_w];
  }
}

template <typename Dtype>
void ArrangeBackLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  const int top_dim = top[0]->count(1);
  const int top_spatial_dim = top[0]->count(2);
  const int top_channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  
  switch (arrangement_) {
  case ArrangeParameter_ArrangementMode_UNION:
    // NOLINT_NEXT_LINE(whitespace/operators)
    UnionModeBackBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff,
        top[0]->num(), channels_, height_, width_,
        region_height_, region_width_, stride_,
        top_channels, top_height, top_width, top_dim, top_spatial_dim,
        bottom_diff);
    break;
  case ArrangeParameter_ArrangementMode_SPLIT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SplitModeBackBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff,
        top[0]->num(), channels_, height_, width_,
        region_height_, region_width_, stride_,
        top_channels, top_height, top_width, top_dim, top_spatial_dim,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown Arrange Mode.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(ArrangeBackLayer);


}  // namespace caffe
