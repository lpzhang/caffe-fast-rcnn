#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_softmax_loss_layer.hpp"
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
void FocalSoftmaxLossLayer<Dtype>::compute_intermediate_values_of_gpu() {
  // compute the corresponding variables
  const int count        = prob_.count();
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* ones_data = ones_.gpu_data();
  Dtype* log_prob_data   = log_prob_.mutable_gpu_data();
  Dtype* power_prob_data = power_prob_.mutable_gpu_data();

  /// log(p_t)
  const int nthreads     = prob_.count();
  const Dtype eps        = Dtype(FLT_MIN); // where FLT_MIN = 1.17549e-38, here u can change it
  // more stable
  // NOLINT_NEXT_LINE(whitespace/operators)
  LogOpGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, log_prob_data, eps);
  /// caffe_gpu_log(count, prob_data, log_prob_data);

  /// alpha * (1 - p_t) ^ (gamma - 1)
  caffe_gpu_sub(count,  ones_data, prob_data, power_prob_data);
  caffe_gpu_powx(count, power_prob_.gpu_data(), gamma_ - 1, power_prob_data);
  caffe_gpu_scal(count, alpha_, power_prob_data);
}

template <typename Dtype>
__global__ void FocalSoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, 
          const Dtype* log_prob_data, 
          const Dtype* power_prob_data, 
          const Dtype* label, 
          const Dtype* label_weight_data, 
          const int bottom_size, 
          const int num, 
          const int dim, 
          const int spatial_dim, 
          const bool has_ignore_label_, 
          const int ignore_label_, 
          Dtype* loss, 
          Dtype* counts) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[index]);
    // Get weight
    Dtype w = 1;
    if (bottom_size == 2) {
      w = label_weight_data[label_value];
    } else {
      w = label_weight_data[index];
    }
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index]   = 0;
      counts[index] = 0;
    } else {
      /*
        FL(p_t) = - alpha * (1 - p_t) ^ gamma * log(p_t)
                = - [alpha * (1 - p_t) ^ (gamma - 1)] * [(1 - p_t)] * [log(p_t)]
      */
      const int ind_t = n * dim + label_value * spatial_dim + s;
      loss[index] = - w * power_prob_data[ind_t] * (1 - prob_data[ind_t]) * log_prob_data[ind_t];
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void FocalSoftmaxLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  // compute all needed values
  compute_intermediate_values_of_gpu();

  // Get label_weight_data
  const Dtype* label_weight_data;
  if (bottom.size() == 2) {
    label_weight_data = label_weight_.gpu_data();
  } else {
    // Get label_weight_data from bottom[2]
    label_weight_data = bottom[2]->gpu_data();
  }

  const Dtype* prob_data       = prob_.gpu_data();
  const Dtype* log_prob_data   = log_prob_.gpu_data();
  const Dtype* power_prob_data = power_prob_.gpu_data();
  const Dtype* label           = bottom[1]->gpu_data();
  const int dim                = prob_.count() / outer_num_;
  const int nthreads           = outer_num_ * inner_num_;

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // the bottom[0] has shape (N,C,H,W) but the FocalSoftmaxLossForwardGPU only 
  // calculate  outer_num_ * inner_num_ times based on the label shape (N,1,H,W)
  // so the loss_data better set 0 at the begining.
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), loss_data);

  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), counts);

  // NOLINT_NEXT_LINE(whitespace/operators)
  FocalSoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, log_prob_data, power_prob_data, label, label_weight_data, 
        bottom.size(), outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
        loss_data, counts);

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;

  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void FocalSoftmaxLossBackwardGPU(const int nthreads, 
          const Dtype* prob_data, 
          const Dtype* log_prob_data, 
          const Dtype* power_prob_data, 
          const Dtype* label, 
          const Dtype* label_weight_data, 
          const Dtype gamma, 
          const int bottom_size, 
          const int num, 
          const int channels,
          const int dim, 
          const int spatial_dim, 
          const bool has_ignore_label_, 
          const int ignore_label_, 
          Dtype* bottom_diff, 
          Dtype* counts) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    // const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int label_value = static_cast<int>(label[index]);
    // Get weight
    Dtype w = 1;
    if (bottom_size <= 2) {
      w = label_weight_data[label_value];
    } else {
      w = label_weight_data[index];
    }

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      // the gradient from FL w.r.t p_t, here ignore the `sign`
      // index of ground-truth label in prob.
      const int ind_t  = n * dim + label_value * spatial_dim + s;
      // alpha * (- gamma * (1 - p_t) ^ (gamma - 1) * p_t * log(p_t) + (1 - p_t) ^ (gamma - 1) * (1 - p_t))
      // - gamma * [alpha * (1 - p_t) ^ (gamma - 1)] * p_t * log(p_t) + [alpha * (1 - p_t) ^ (gamma - 1)] * (1 - p_t)
      Dtype grad = 0 - gamma * power_prob_data[ind_t] * prob_data[ind_t] * log_prob_data[ind_t]
                    + power_prob_data[ind_t] * (1 - prob_data[ind_t]);
      // Weight each channel
      for (int c = 0; c < channels; ++c) {
        int ind_k = n * dim + c * spatial_dim + s;
        if(c == label_value) {
          // if t == k, (here t,k are refered for derivative of softmax), grad * (p_t -1)
          bottom_diff[ind_k] = w * grad * (prob_data[ind_t] - 1);
        } else {
          // if t != k, (here t,k are refered for derivative of softmax), grad * p_k
          bottom_diff[ind_k] = w * grad * prob_data[ind_k];
        }
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void FocalSoftmaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff     = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label     = bottom[1]->gpu_data();
    const int channels     = bottom[0]->shape(softmax_axis_);
    const int dim          = prob_.count() / outer_num_;
    const int nthreads     = outer_num_ * inner_num_;

    // Get label_weight_data
    const Dtype* label_weight_data;
    if (bottom.size() == 2) {
      label_weight_data = label_weight_.gpu_data();
    } else {
      // Get label_weight_data from bottom[2]
      label_weight_data = bottom[2]->gpu_data();
    }

    // intermidiate  
    const Dtype* log_prob_data   = log_prob_.gpu_data();
    const Dtype* power_prob_data = power_prob_.gpu_data();

    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.), counts);

    // NOLINT_NEXT_LINE(whitespace/operators)
    FocalSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, log_prob_data, power_prob_data, label, label_weight_data, gamma_, 
          bottom.size(), outer_num_, channels, dim, inner_num_, has_ignore_label_, ignore_label_,
          bottom_diff, counts);

    // Only launch another CUDA kernel if we actually need the count of valid outputs.
    Dtype valid_count = -1;
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FocalSoftmaxLossLayer);

}  // namespace caffe
