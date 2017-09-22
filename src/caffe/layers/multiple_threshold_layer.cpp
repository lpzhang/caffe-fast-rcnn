#include <vector>

#include "caffe/layers/multiple_threshold_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultipleThresholdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const MultipleThresholdParameter& multiple_threshold_param = this->layer_param_.multiple_threshold_param();
  point_size_ = multiple_threshold_param.threshold_point_size();
  int value_size = multiple_threshold_param.threshold_value_size();
  CHECK_EQ(1, value_size - point_size_)
        << "value_size - point_size must equal to 1";
  threshold_point_.Reshape(point_size_, 1, 1, 1);
  threshold_value_.Reshape(value_size, 1, 1, 1);
  Dtype* threshold_point_data = threshold_point_.mutable_cpu_data();
  int* threshold_value_data = threshold_value_.mutable_cpu_data();
  for (int i = 0; i < point_size_; ++i) {
    threshold_point_data[i] = multiple_threshold_param.threshold_point(i);
    // LOG(INFO) << threshold_point_data[i];
  }
  for (int i = 0; i < value_size; ++i) {
    threshold_value_data[i] = multiple_threshold_param.threshold_value(i);
    // LOG(INFO) << threshold_value_data[i];
  }
}

template <typename Dtype>
void MultipleThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // Init top
  caffe_set(top[0]->count(), Dtype(0), top_data);
  
  const int count = bottom[0]->count();
  const Dtype* threshold_point_data = threshold_point_.cpu_data();
  const int* threshold_value_data = threshold_value_.cpu_data();
  for (int i = 0; i < count; ++i) {
    // top_data[i] = (bottom_data[i] > threshold_) ? Dtype(1) : Dtype(0);
    int value_index = -1;
    for (int j = 0; j < point_size_; ++j) {
      if (bottom_data[i] >= threshold_point_data[j]) {
        value_index = j;
      }
    }
    value_index += 1;
    top_data[i] = threshold_value_data[value_index];
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(MultipleThresholdLayer, Forward);
#endif

INSTANTIATE_CLASS(MultipleThresholdLayer);
REGISTER_LAYER_CLASS(MultipleThreshold);

}  // namespace caffe
