#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/arrange_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UnionModeForward(const int nthreads, const Dtype* const bottom_data,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width,
    const int top_dim, const int top_spatial_dim, const int stride,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    int region_index_h = h % stride;
    int region_index_w = w % stride;
    int region_h = h / stride + region_index_h * region_height;
    int region_w = w / stride + region_index_w * region_width;

    Dtype* const top_slice = top_data + n * top_dim + c * top_spatial_dim;
    top_slice[region_h * region_width + region_w] = bottom_data[index];
  }
}

template <typename Dtype>
__global__ void SplitModeForward(const int nthreads, const Dtype* const bottom_data,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width,
    const int top_dim, const int top_spatial_dim, const int stride, const int region_dist,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    int region_index_h = h % stride;
    int region_index_w = w % stride;
    int region_h = h / stride;
    int region_w = w / stride;
    int region_index = region_index_h * stride + region_index_w;

    // int region_offset = region_index * region_dist;
    Dtype* const top_slice = top_data + n * top_dim + region_index * region_dist + c * top_spatial_dim;
    top_slice[region_h * region_width + region_w] = bottom_data[index];
  }
}

template <typename Dtype>
void ArrangeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);

  const int count = bottom[0]->count();
  const int top_dim = top[0]->count(1);
  const int top_spatial_dim = top[0]->count(2);
  const int region_dist = channels_ * spatial_dim;
 
  switch (arrangement_) {
  case ArrangeParameter_ArrangementMode_UNION:
    // NOLINT_NEXT_LINE(whitespace/operators)
    UnionModeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,
        bottom[0]->num(), channels_, height_, width_,
        region_height_, region_width_,
        top_dim, top_spatial_dim, stride_,
        top_data);
    break;
  case ArrangeParameter_ArrangementMode_SPLIT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SplitModeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,
        bottom[0]->num(), channels_, height_, width_,
        region_height_, region_width_,
        top_dim, top_spatial_dim, stride_, region_dist,
        top_data);
    break;
  default:
    LOG(FATAL) << "Unknown Arrange Mode.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void UnionModeBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width,
    const int top_dim, const int top_spatial_dim, const int stride,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    int region_index_h = h % stride;
    int region_index_w = w % stride;
    int region_h = h / stride + region_index_h * region_height;
    int region_w = w / stride + region_index_w * region_width;

    const Dtype* const top_slice = top_diff + n * top_dim + c * top_spatial_dim;
    bottom_diff[index] = top_slice[region_h * region_width + region_w];
  }
}

template <typename Dtype>
__global__ void SplitModeBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int region_height, const int region_width,
    const int top_dim, const int top_spatial_dim, const int stride, const int region_dist,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    int region_index_h = h % stride;
    int region_index_w = w % stride;
    int region_h = h / stride;
    int region_w = w / stride;
    int region_index = region_index_h * stride + region_index_w;

    const Dtype* const top_slice = top_diff + n * top_dim + region_index * region_dist + c * top_spatial_dim;
    bottom_diff[index] = top_slice[region_h * region_width + region_w];
  }
}

template <typename Dtype>
void ArrangeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
  const int region_dist = channels_ * spatial_dim;
  
  switch (arrangement_) {
  case ArrangeParameter_ArrangementMode_UNION:
    // NOLINT_NEXT_LINE(whitespace/operators)
    UnionModeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff,
        top[0]->num(), channels_, height_, width_,
        region_height_, region_width_,
        top_dim, top_spatial_dim, stride_,
        bottom_diff);
    break;
  case ArrangeParameter_ArrangementMode_SPLIT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SplitModeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff,
        top[0]->num(), channels_, height_, width_,
        region_height_, region_width_,
        top_dim, top_spatial_dim, stride_, region_dist,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown Arrange Mode.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(ArrangeLayer);


}  // namespace caffe
