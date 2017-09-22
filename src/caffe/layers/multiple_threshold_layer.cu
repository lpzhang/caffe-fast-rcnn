#include <vector>

#include "caffe/layers/multiple_threshold_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultipleThresholdForward(const int n, const int point_size, const Dtype* threshold_point, const int* threshold_value,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    // out[index] = in[index] > threshold ? 1 : 0;
    int value_index = -1;
    for (int i = 0; i < point_size; ++i) {
      if (in[index] >= threshold_point[i]) {
        value_index = i;
      }
    }
    value_index += 1;
    out[index] = threshold_value[value_index];
  }
}

template <typename Dtype>
void MultipleThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // Init top
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);

  const int count = bottom[0]->count();
  const Dtype* threshold_point_data = threshold_point_.gpu_data();
  const int* threshold_value_data = threshold_value_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultipleThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, point_size_, threshold_point_data, threshold_value_data, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FORWARD(MultipleThresholdLayer);


}  // namespace caffe