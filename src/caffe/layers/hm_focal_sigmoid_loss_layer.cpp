#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/hm_focal_sigmoid_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HMFocalSigmoidLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // sigmoid laye setup
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter sigmoid_param(this->layer_param_);
  sigmoid_param.set_type("Sigmoid");
  sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&prob_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  // ignore label
  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_   = this->layer_param_.loss_param().ignore_label();
  }

  // normalization
  if (!this->layer_param_.loss_param().has_normalization() &&
       this->layer_param_.loss_param().has_normalize())
  {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  // focal sigmoid loss parameter
  HMFocalSigmoidLossParameter hm_focal_sigmoid_loss_param = this->layer_param_.hm_focal_sigmoid_loss_param();
  alpha_  = hm_focal_sigmoid_loss_param.alpha();
  beta_   = hm_focal_sigmoid_loss_param.beta();
  gamma_  = hm_focal_sigmoid_loss_param.gamma();
  radius_ = hm_focal_sigmoid_loss_param.radius();
  LOG(INFO) << "alpha: " << alpha_;
  LOG(INFO) << "beta: "  << beta_;
  LOG(INFO) << "gamma: " << gamma_;
  LOG(INFO) << "radius: " << radius_;
  CHECK_GT(alpha_, 0) << "alpha must be larger than zero";
  CHECK_GE(beta_,  0) << "beta must be larger than or equal to zero";
  CHECK_GE(gamma_, 0) << "gamma must be larger than or equal to zero";
  CHECK_GT(radius_, 0) << "radius must be larger than zero";
}

template <typename Dtype>
void HMFocalSigmoidLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  LossLayer<Dtype>::Reshape(bottom, top);

  // sigmoid laye reshape
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << 
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  if (bottom.size() == 3) {
    CHECK_EQ(bottom[0]->count(), bottom[2]->count()) << 
        "SIGMOID_CROSS_ENTROPY_LOSS layer weight must be the same count";
  }
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  outer_num_ = bottom[ 0 ]->shape(0);  // batch size
  inner_num_ = bottom[ 0 ]->count(1);  // instance size: |output| == |target|

  // sigmoid output
  if (top.size() >= 2) {
    top[1]->ReshapeLike(*bottom[0]);
  }

  // log(p_t)
  log_prob_.ReshapeLike(*bottom[0]);
  CHECK_EQ(prob_.count(), log_prob_.count());
  // log(1 - p_t)
  log_neg_prob_.ReshapeLike(*bottom[0]);
  CHECK_EQ(prob_.count(), log_neg_prob_.count());
  // alpha * p_t ^ gamma
  power_prob_.ReshapeLike(*bottom[0]);
  CHECK_EQ(prob_.count(), power_prob_.count());
  // alpha * (1 - p_t) ^ gamma
  power_neg_prob_.ReshapeLike(*bottom[0]);
  CHECK_EQ(prob_.count(), power_neg_prob_.count());
  // (1 − y_t) ^ beta
  power_penalty_.ReshapeLike(*bottom[0]);
  CHECK_EQ(prob_.count(), power_penalty_.count());
  // 1
  ones_.ReshapeLike(*bottom[0]);
  CHECK_EQ(prob_.count(), ones_.count());
  caffe_set(prob_.count(), Dtype(1), ones_.mutable_cpu_data());
}

template <typename Dtype>
Dtype HMFocalSigmoidLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count)
{
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void HMFocalSigmoidLossLayer<Dtype>::compute_intermediate_values_of_cpu(const Dtype* target) {
  // compute the corresponding variables
  const int count        = prob_.count();
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* ones_data = ones_.cpu_data();
  Dtype* log_prob_data       = log_prob_.mutable_cpu_data();
  Dtype* log_neg_prob_data   = log_neg_prob_.mutable_cpu_data();
  Dtype* power_prob_data     = power_prob_.mutable_cpu_data();
  Dtype* power_neg_prob_data = power_neg_prob_.mutable_cpu_data();
  Dtype* power_penalty_data  = power_penalty_.mutable_cpu_data();

  /// log(p_t) and log(1 - p_t), where FLT_MIN = 1.17549e-38, here u can change it
  const Dtype eps = Dtype(FLT_MIN);
  // more stable than caffe_log(count,  prob_data, log_prob_data);
  for(int i = 0; i < count; i++) {
    log_prob_data[i]     = log(std::max(prob_data[i], eps));
    log_neg_prob_data[i] = log(std::max(Dtype(1.0) - prob_data[i], eps));
  }

  /// alpha * p_t ^ gamma
  caffe_powx(count, prob_data, gamma_, power_prob_data);
  caffe_scal(count, alpha_, power_prob_data);

  /// alpha * (1 - p_t) ^ gamma
  caffe_sub(count,  ones_data, prob_data, power_neg_prob_data);
  caffe_powx(count, power_neg_prob_.cpu_data(), gamma_, power_neg_prob_data);
  caffe_scal(count, alpha_, power_neg_prob_data);

  /// (1 - y_t) ^ beta
  caffe_sub(count,  ones_data, target, power_penalty_data);
  caffe_powx(count, power_penalty_.cpu_data(), beta_, power_penalty_data);
}

template <typename Dtype>
void HMFocalSigmoidLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // The forward pass computes the sigmoid prob values.
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  // compute all needed values
  compute_intermediate_values_of_cpu(bottom[1]->cpu_data());
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* log_prob_data       = log_prob_.cpu_data();
  const Dtype* log_neg_prob_data   = log_neg_prob_.cpu_data();
  const Dtype* power_prob_data     = power_prob_.cpu_data();
  const Dtype* power_neg_prob_data = power_neg_prob_.cpu_data();
  const Dtype* power_penalty_data  = power_penalty_.cpu_data();

  // Compute the loss
  const int count = bottom[0]->count();
  Dtype loss = 0;
  Dtype num_pos = Dtype(0.0);

  if (bottom.size() == 2) {
    for (int i = 0; i < count; ++i) {
      if (fabs(Dtype(1.0) - target[i]) <= radius_) {
        // positive location target value equal to 1
        // - [ alpha * (1 - p_t) ^ gamma ] * [ log(p_t) ]
        loss -= power_neg_prob_data[i] * log_prob_data[i];
        num_pos += 1.0;
      } else {
        // negative location target value less than 1
        // - [ alpha * p_t ^ gamma ] * [ log(1 - p_t) ] * [ (1 - y_t) ^ beta ]
        loss -= power_prob_data[i] * log_neg_prob_data[i] * power_penalty_data[i];
      }
    }
    // top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  } else if (bottom.size() == 3) {
    const Dtype* weights = bottom[2]->cpu_data();
    // Dtype weight_sum = Dtype(0.0);
    for (int i = 0; i < count; ++i) {
      if (fabs(Dtype(1.0) - target[i]) <= radius_) {
        // positive location target value equal to 1
        // - [ weights ] * [ alpha * (1 - p_t) ^ gamma ] * [ log(p_t) ]
        loss -= weights[i] * power_neg_prob_data[i] * log_prob_data[i];
        // num_pos += weights[i];
        num_pos += 1.0;
      } else {
        // negative location target value less than 1
        // - [ weights ] * [ alpha * p_t ^ gamma ] * [ log(1 - p_t) ] * [ (1 - y_t) ^ beta ]
        loss -= weights[i] * power_prob_data[i] * log_neg_prob_data[i] * power_penalty_data[i];
      }
      // weight_sum += weights[i];
    }
    // if (weight_sum > Dtype(0.0)) {
    //   top[0]->mutable_cpu_data()[0] = loss / weight_sum;
    // } else {
    //   top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    // }
  }
  // Normalization
  if (num_pos > 0) {
    top[0]->mutable_cpu_data()[0] = loss / num_pos;
  } else {
    top[0]->mutable_cpu_data()[0] = loss;
  }
  // sigmoid output
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void HMFocalSigmoidLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {
    // data
    Dtype* bottom_diff     = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* target    = bottom[1]->cpu_data();
    // intermidiate
    const Dtype* log_prob_data       = log_prob_.cpu_data();
    const Dtype* log_neg_prob_data   = log_neg_prob_.cpu_data();
    const Dtype* power_prob_data     = power_prob_.cpu_data();
    const Dtype* power_neg_prob_data = power_neg_prob_.cpu_data();
    const Dtype* power_penalty_data  = power_penalty_.cpu_data();

    const int count = bottom[0]->count();
    const Dtype loss_weight = top[0]->cpu_diff()[0]; // Scale down gradient
    Dtype num_pos = Dtype(0.0);
    
    if (bottom.size() == 2) {
      for (int i = 0; i < count; ++i) {
        if (fabs(Dtype(1.0) - target[i]) <= radius_) {
          // positive location target value equal to 1
          // - [ alpha * (1 - p_t) ^ gamma ] * [ - gamma * p_t * log(p_t) + (1 - p_t) ]
          bottom_diff[i] = 0 - power_neg_prob_data[i] 
                              * (0 - gamma_ * prob_data[i] * log_prob_data[i] + (1 - prob_data[i]));
          num_pos += 1.0;
        } else {
          // negative location target less than 1
          // - [ alpha * p_t ^ gamma ] * [ gamma * (1 - p_t) * log(1 - p_t) - p_t ] * [ (1 - y_t) ^ beta ]
          bottom_diff[i] = 0 - power_prob_data[i] 
                              * (gamma_ * (1 - prob_data[i]) * log_neg_prob_data[i] - prob_data[i]) 
                              * power_penalty_data[i];
        }
      }
      // Scale down gradient
      // caffe_scal(count, loss_weight / get_normalizer(normalization_, count), bottom_diff);
    } else if (bottom.size() == 3) {
      const Dtype* weights = bottom[2]->cpu_data();
      // Dtype weight_sum = 0.0;
      for (int i = 0; i < count; ++i) {
        // bottom_diff[i] *= weights[i];
        if (fabs(Dtype(1.0) - target[i]) <= radius_) {
          // positive location target value equal to 1
          // - [ weights ] * [ alpha * (1 - p_t) ^ gamma ] * [ - gamma * p_t * log(p_t) + (1 - p_t) ]
          bottom_diff[i] = 0 - weights[i] * power_neg_prob_data[i] 
                              * (0 - gamma_ * prob_data[i] * log_prob_data[i] + (1 - prob_data[i]));
          // num_pos += weights[i];
          num_pos += 1.0;
        } else {
          // negative location target less than 1
          // - [ weights ] * [ alpha * p_t ^ gamma ] * [ gamma * (1 - p_t) * log(1 - p_t) - p_t ] * [ (1 - y_t) ^ beta ]
          bottom_diff[i] = 0 - weights[i] * power_prob_data[i] 
                              * (gamma_ * (1 - prob_data[i]) * log_neg_prob_data[i] - prob_data[i]) 
                              * power_penalty_data[i];
        }
        // weight_sum += weights[i];
      }
      // Scale down gradient
      // if (weight_sum > 0.0) {
      //   caffe_scal(count, loss_weight / weight_sum, bottom_diff);
      // } else {
      //   caffe_scal(count, loss_weight / get_normalizer(normalization_, count), bottom_diff);
      // }
    }
    // Scale down gradient
    if (num_pos > 0) {
      caffe_scal(count, loss_weight / num_pos, bottom_diff);
    } else {
      caffe_scal(count, loss_weight, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HMFocalSigmoidLossLayer);
#endif

INSTANTIATE_CLASS(HMFocalSigmoidLossLayer);
REGISTER_LAYER_CLASS(HMFocalSigmoidLoss);

}  // namespace caffe
