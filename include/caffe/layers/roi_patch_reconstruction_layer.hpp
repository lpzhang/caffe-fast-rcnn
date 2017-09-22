#ifndef CAFFE_ROI_PATCH_RECONSTRUCTION_LAYER_HPP_
#define CAFFE_ROI_PATCH_RECONSTRUCTION_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Changes the spatial resolution by bi-linear interpolation.
 *        The target size is specified in terms of pixels. 
 *        The start and end pixels of the input are mapped to the start
 *        and end pixels of the output.
 */
template <typename Dtype>
class ROIPatchReconstructionLayer : public Layer<Dtype> {
 public:
  explicit ROIPatchReconstructionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPatchReconstruction"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int num_;
  int channels_;
  int height_in_;
  int width_in_;
  int height_out_;
  int width_out_;

  Blob<Dtype> in_;  // cached for in_
  Blob<Dtype> out_;  // cached for out_
};

}  // namespace caffe

#endif  // CAFFE_ROI_PATCH_RECONSTRUCTION_LAYER_HPP_
