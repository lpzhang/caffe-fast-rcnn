#ifndef CAFFE_MULTIPLE_THRESHOLD_LAYER_HPP_
#define CAFFE_MULTIPLE_THRESHOLD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/neuron_layers.hpp"

namespace caffe {
template <typename Dtype>
class MultipleThresholdLayer : public NeuronLayer<Dtype> {
 public:
  explicit MultipleThresholdLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultipleThreshold"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int point_size_;
  Blob<Dtype> threshold_point_;  // cached for threshold_point_
  Blob<int> threshold_value_;  // cached for threshold_value_
};

}  // namespace caffe

#endif  // CAFFE_MULTIPLE_THRESHOLD_LAYER_HPP_
