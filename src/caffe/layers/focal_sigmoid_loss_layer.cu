#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_sigmoid_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void LogOpGPU(const int nthreads,
          const Dtype* in, Dtype* out, const Dtype eps)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = log(max(in[index], eps));
  }
}

template <typename Dtype>
void FocalSigmoidLossLayer<Dtype>::compute_intermediate_values_of_gpu(const Dtype* target) {
  // compute the corresponding variables
  const int count        = prob_.count();
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* ones_data = ones_.gpu_data();
  Dtype* log_prob_data       = log_prob_.mutable_gpu_data();
  Dtype* log_neg_prob_data   = log_neg_prob_.mutable_gpu_data();
  Dtype* power_prob_data     = power_prob_.mutable_gpu_data();
  Dtype* power_neg_prob_data = power_neg_prob_.mutable_gpu_data();
  Dtype* power_penalty_data  = power_penalty_.mutable_gpu_data();

  /// log(p_t)
  const int nthreads     = prob_.count();
  const Dtype eps        = Dtype(FLT_MIN); // where FLT_MIN = 1.17549e-38, here u can change it
  // more stable
  // NOLINT_NEXT_LINE(whitespace/operators)
  LogOpGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, log_prob_data, eps);
  /// caffe_gpu_log(count,  prob_data, log_prob_data);
  
  /// log(1- p_t)
  caffe_gpu_sub(count, ones_data, prob_data, log_neg_prob_data);
  LogOpGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, log_neg_prob_.gpu_data(), log_neg_prob_data, eps);

  /// alpha * p_t ^ gamma
  caffe_gpu_powx(count, prob_data, gamma_, power_prob_data);
  caffe_gpu_scal(count, alpha_, power_prob_data);

  /// alpha * (1 - p_t) ^ gamma
  caffe_gpu_sub(count,  ones_data, prob_data, power_neg_prob_data);
  caffe_gpu_powx(count, power_neg_prob_.gpu_data(), gamma_, power_neg_prob_data);
  caffe_gpu_scal(count, alpha_, power_neg_prob_data);

  /// (1 - y_t) ^ beta
  caffe_gpu_sub(count,  ones_data, target, power_penalty_data);
  caffe_gpu_powx(count, power_penalty_.gpu_data(), beta_, power_penalty_data);
}

template <typename Dtype>
__global__ void FocalSigmoidLossForwardGPU(const int nthreads,
          const Dtype* log_prob_data, 
          const Dtype* log_neg_prob_data, 
          const Dtype* power_prob_data, 
          const Dtype* power_neg_prob_data, 
          const Dtype* power_penalty_data, 
          const Dtype* target, 
          Dtype* loss, 
          Dtype* counts,
          const Dtype eps) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (fabs(Dtype(1.0) - target[index]) < eps) {
      // positive location target value equal to 1
      // - [ alpha * (1 - p_t) ^ gamma ] * [ log(p_t) ]
      loss[index] = 0 - power_neg_prob_data[index] * log_prob_data[index];
      counts[index] = 1;
    } else {
      // negative location target value less than 1
      // - [ alpha * p_t ^ gamma ] * [ log(1 - p_t) ] * [ (1 - y_t) ^ beta ]
      loss[index] = 0 - power_prob_data[index] * log_neg_prob_data[index] * power_penalty_data[index];
      counts[index] = 0;
    }
  }
}

template <typename Dtype>
void FocalSigmoidLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // The forward pass computes the sigmoid prob values.
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  // compute all needed values
  compute_intermediate_values_of_gpu(bottom[1]->gpu_data());

  // const Dtype* prob_data       = prob_.gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  const Dtype* log_prob_data       = log_prob_.gpu_data();
  const Dtype* log_neg_prob_data   = log_neg_prob_.gpu_data();
  const Dtype* power_prob_data     = power_prob_.gpu_data();
  const Dtype* power_neg_prob_data = power_neg_prob_.gpu_data();
  const Dtype* power_penalty_data  = power_penalty_.gpu_data();
  const int nthreads = prob_.count();
  const Dtype eps = Dtype(1e-4);

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  // number of positive location (target value equal to 1)
  Dtype* counts = prob_.mutable_gpu_diff();

  // NOLINT_NEXT_LINE(whitespace/operators)
  FocalSigmoidLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, log_prob_data, log_neg_prob_data, power_prob_data,
        power_neg_prob_data, power_penalty_data, target, loss_data, counts, eps);

  // weights in bottom[2] if any
  if (bottom.size() == 3) {
    caffe_gpu_mul(nthreads, bottom[2]->gpu_data(), bottom[0]->gpu_diff(), loss_data);
    caffe_gpu_mul(nthreads, bottom[2]->gpu_data(), prob_->gpu_diff(), counts);
  }
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  // Dtype valid_count = -1;
  // // Only launch another CUDA kernel if we actually need the count of valid
  // // outputs.
  // if (normalization_ == LossParameter_NormalizationMode_VALID &&
  //     has_ignore_label_) {
  //   caffe_gpu_asum(nthreads, counts, &valid_count);
  // }
  // top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, valid_count);
  // if (bottom.size() == 2) {
  //   top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, nthreads);
  // } else if (bottom.size() == 3) {
  //   Dtype weight_sum;
  //   caffe_gpu_asum(nthreads, bottom[2]->gpu_data(), &weight_sum);
  //   if (weight_sum > Dtype(0.0)) {
  //     top[0]->mutable_cpu_data()[0] = loss / weight_sum;
  //   } else {
  //     top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, nthreads);
  //   }
  // }
  Dtype num_pos = 0;
  caffe_gpu_asum(nthreads, counts, &num_pos);
  if (num_pos > eps) {
    top[0]->mutable_cpu_data()[0] = loss / num_pos;
  } else {
    top[0]->mutable_cpu_data()[0] = loss;
  }

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void FocalSigmoidLossBackwardGPU(const int nthreads, 
          const Dtype* prob_data,
          const Dtype* log_prob_data, 
          const Dtype* log_neg_prob_data, 
          const Dtype* power_prob_data, 
          const Dtype* power_neg_prob_data, 
          const Dtype* power_penalty_data, 
          const Dtype* target, 
          Dtype* bottom_diff, 
          Dtype* counts,
          const Dtype gamma_,
          const Dtype eps) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (fabs(Dtype(1.0) - target[index]) < eps) {
      // positive location target value equal to 1
      // - [ alpha * (1 - p_t) ^ gamma ] * [ - gamma * p_t * log(p_t) + (1 - p_t) ]
      bottom_diff[index] = 0 - power_neg_prob_data[index] 
                              * (0 - gamma_ * prob_data[index] * log_prob_data[index] + (1 - prob_data[index]));
      counts[index] = 1;
    } else {
      // negative location target less than 1
      // - [ alpha * p_t ^ gamma ] * [ gamma * (1 - p_t) * log(1 - p_t) - p_t ] * [ (1 - y_t) ^ beta ]
      bottom_diff[index] = 0 - power_prob_data[index] 
                              * (gamma_ * (1 - prob_data[index]) * log_neg_prob_data[index] - prob_data[index]) 
                              * power_penalty_data[index];
      counts[index] = 0;
    }
  }
}

template <typename Dtype>
void FocalSigmoidLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff     = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    // const Dtype* top_data  = top[0]->gpu_data();
    const Dtype* target    = bottom[1]->gpu_data();
    const int nthreads     = prob_.count();
    const Dtype eps        = Dtype(1e-4);

    // intermidiate  
    const Dtype* log_prob_data   = log_prob_.gpu_data();
    const Dtype* log_neg_prob_data   = log_neg_prob_.gpu_data();
    const Dtype* power_prob_data = power_prob_.gpu_data();
    const Dtype* power_neg_prob_data = power_neg_prob_.gpu_data();
    const Dtype* power_penalty_data = power_penalty_.gpu_data();

    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();

    // NOLINT_NEXT_LINE(whitespace/operators)
    FocalSigmoidLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, log_prob_data, log_neg_prob_data, power_prob_data,
          power_neg_prob_data, power_penalty_data, target, bottom_diff, counts, gamma_, eps);

    // weights in bottom[2] if any
    if (bottom.size() == 3) {
      caffe_gpu_mul(nthreads, bottom[2]->gpu_data(), bottom[0]->gpu_diff(), bottom_diff);
      caffe_gpu_mul(nthreads, bottom[2]->gpu_data(), prob_->gpu_diff(), counts);
    }
    // // Only launch another CUDA kernel if we actually need the count of valid outputs.
    // Dtype valid_count = -1;
    // if (normalization_ == LossParameter_NormalizationMode_VALID &&
    //     has_ignore_label_) {
    //   caffe_gpu_asum(nthreads, counts, &valid_count);
    // }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    // if (bottom.size() == 2) {
    //   caffe_gpu_scal(nthreads, loss_weight / get_normalizer(normalization_, nthreads), bottom_diff);
    // } else if (bottom.size() == 3) {
    //   Dtype weight_sum;
    //   caffe_gpu_asum(nthreads, bottom[2]->gpu_data(), &weight_sum);
    //   if (weight_sum > Dtype(0.0)) {
    //     caffe_gpu_scal(nthreads, loss_weight / weight_sum, bottom_diff);
    //   } else {
    //     caffe_gpu_scal(nthreads, loss_weight / get_normalizer(normalization_, nthreads), bottom_diff);
    //   }
    // }
    Dtype num_pos = 0;
    caffe_gpu_asum(nthreads, counts, &num_pos);
    if (num_pos > eps) {
      caffe_gpu_scal(nthreads, loss_weight / num_pos, bottom_diff);
    } else {
      caffe_gpu_scal(nthreads, loss_weight, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FocalSigmoidLossLayer);

}  // namespace caffe
