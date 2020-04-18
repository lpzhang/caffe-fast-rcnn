#ifndef CAFFE_HM_FOCAL_SIGMOID_LOSS_LAYER_HPP_
#define CAFFE_HM_FOCAL_SIGMOID_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * A variant of focal loss ("Focal Loss for Dense Object Detection")
 * For details, see "CornerNet: Detecting Objects as Paired Keypoints"
 */
template <typename Dtype>
class HMFocalSigmoidLossLayer : public LossLayer<Dtype> {
 public:
  explicit HMFocalSigmoidLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HMFocalSigmoidLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  void compute_intermediate_values_of_cpu(const Dtype* target);
  void compute_intermediate_values_of_gpu(const Dtype* target);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<Layer<Dtype> > sigmoid_layer_;
  /// prob stores the output probability predictions from the SigmoidLayer.
  Blob<Dtype> prob_;            // sigmoid output p_t
  Blob<Dtype> log_prob_;        // log(p_t)
  Blob<Dtype> log_neg_prob_;    // log(1 - p_t)
  Blob<Dtype> power_prob_;      // alpha * p_t ^ gamma
  Blob<Dtype> power_neg_prob_;  // alpha * (1 - p_t) ^ gamma
  Blob<Dtype> power_penalty_;   // (1 - y_t) ^ beta reduces the penalty around the ground truth locations.
  Blob<Dtype> ones_;            // 1

  /// bottom vector holder used in call to the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder used in call to the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;
  ///
  // FocalSigmoidLossParameter_Type type_;

  Dtype alpha_, beta_, gamma_, radius_;
  int outer_num_, inner_num_;
};

}  // namespace caffe

#endif  // CAFFE_HM_FOCAL_SIGMOID_LOSS_LAYER_HPP_
