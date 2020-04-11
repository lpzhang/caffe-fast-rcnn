#include <vector>

#include "caffe/layers/kp2_smooth_l1_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = 0.5 * val * val * sigma2;
    } else {
      out[index] = abs_val - 0.5 / sigma2;
    }
  }
}

template <typename Dtype>
__global__ void KeypointForward(const int nthreads,
    const Dtype* prob_data, const Dtype* label, const Dtype* mask, const Dtype* iloc,
    Dtype* diff_,
    const int channels, const int num_objs,
    const int prob_spatial_dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the n, c, and s of the index in label (N, C, num_objs)
    const int n = index / channels / num_objs;
    const int c = index / num_objs;
    const int s = index % num_objs;
    // calculate the corresponding index in prob_data (N, C, H, W)
    // mask (N, num_objs), iloc (N, num_objs)
    const int mask_val = static_cast<int>(mask[n * num_objs + s]);
    if (mask_val > 0) {
      const int spatial_index = static_cast<int>(iloc[n * num_objs + s]);
      const int prob_index = n * channels * prob_spatial_dim + c * prob_spatial_dim + spatial_index;
      diff_[index] = prob_data[prob_index] - label[index];
    }
  }
}

template <typename Dtype>
void KP2SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // prediction (N, C, H, W), 
  // groundtruth(N, C, num_objs),
  // binary indicator of objects existence: mask (N, num_objs)
  // spatial location index of objects    : iloc (N, num_objs)
  const Dtype* prob_data = bottom[0]->gpu_data();
  const Dtype* label     = bottom[1]->gpu_data();
  const Dtype* mask      = bottom[2]->gpu_data();
  const Dtype* iloc      = bottom[3]->gpu_data();
  // count, channels, and num_objs of the label (N, C, num_objs)
  const int count    = bottom[1]->count();
  const int channels = bottom[1]->shape(1);
  const int num_objs = bottom[1]->shape(2);
  // spatial_dim (H*W) of the prob_data 
  const int prob_spatial_dim = bottom[0]->count(2);

  // Compute the keypoint diff_ w_in * (b0 - b1)
  // NOLINT_NEXT_LINE(whitespace/operators)
  KeypointForward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, prob_data, label, mask, iloc,
      diff_.mutable_gpu_data(),
      channels, num_objs, prob_spatial_dim);
  // caffe_gpu_sub(
  //     count,
  //     bottom[0]->gpu_data(),
  //     bottom[1]->gpu_data(),
  //     diff_.mutable_gpu_data());    // d := b0 - b1
  // inside weights bottom[4]->gpu_data() (N, C, num_objs)
  if (has_weights_) {
    // apply "inside" weights
    caffe_gpu_mul(
        count,
        bottom[4]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w_in * (b0 - b1)
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;
  // outside weights bottom[5]->gpu_data() (N, C, num_objs)
  if (has_weights_) {
    // apply "outside" weights
    caffe_gpu_mul(
        count,
        bottom[5]->gpu_data(),
        errors_.gpu_data(),
        errors_.mutable_gpu_data());  // d := w_out * SmoothL1(w_in * (b0 - b1))
  }

  Dtype loss;
  caffe_gpu_dot(count, ones_.gpu_data(), errors_.gpu_data(), &loss);
  // calculate the total numbers of positive objects by summing mask (N, num_objs)
  Dtype num_pos = 0;
  caffe_gpu_asum(bottom[2]->count(), bottom[2]->gpu_data(), &num_pos);
  top[0]->mutable_cpu_data()[0] = loss / (num_pos + Dtype(1e-4));
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
  //       = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = sigma2 * val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}

template <typename Dtype>
__global__ void KeypointBackward(const int nthreads,
    const Dtype* diff_, const Dtype* mask, const Dtype* iloc,
    Dtype* bottom_diff,
    const int channels, const int num_objs,
    const int prob_spatial_dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // the n, c, and s of the index in diff_ (N, C, num_objs)
    const int n = index / channels / num_objs;
    const int c = index / num_objs;
    const int s = index % num_objs;
    // calculate the corresponding index in bottom_diff (N, C, H, W)
    const int mask_val = static_cast<int>(mask[n * num_objs + s]);
    const int spatial_index = static_cast<int>(iloc[n * num_objs + s]);
    const int prob_index = n * channels * prob_spatial_dim + c * prob_spatial_dim + spatial_index;
    if (mask_val > 0) {
      bottom_diff[prob_index] = diff_[index]
    } else {
      bottom_diff[prob_index] = 0;
    }
  }
}

template <typename Dtype>
void KP2SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    // mask (N, num_objs), iloc (N, num_objs)
    const Dtype* mask  = bottom[2]->gpu_data();
    const Dtype* iloc  = bottom[3]->gpu_data();
    // count, channels, and num_objs of the diff_ (N, C, num_objs)
    const int count    = diff_.count();
    const int channels = diff_.shape(1);
    const int num_objs = diff_.shape(2);
    // spatial_dim (H*W) of the prediction (N, C, H, W)
    const int prob_spatial_dim = bottom[0]->count(2);

    // after forwards, diff_ holds w_in * (b0 - b1)
    // NOLINT_NEXT_LINE(whitespace/operators)
    SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diff_.gpu_data(), diff_.mutable_gpu_data(), sigma2_);
    CUDA_POST_KERNEL_CHECK;
    
    // Scale diff_ first, since diff_ and weights have the same dimension (N, C, num_objs)
    if (has_weights_) {
      // Scale by "inside" weight bottom[4] with shape (N, C, num_objs)
      caffe_gpu_mul(count, bottom[4]->gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
      // Scale by "outside" weight bottom[5] with shape (N, C, num_objs)
      caffe_gpu_mul(count, bottom[5]->gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
    }
    // Scale gradient
    // calculate the total numbers of positive objects by summing mask (N, num_objs)
    Dtype num_pos = 0;
    caffe_gpu_asum(bottom[2]->count(), bottom[2]->gpu_data(), &num_pos);
    const Dtype loss_weight = top[0]->cpu_diff()[0] / (num_pos + Dtype(1e-4));      
    caffe_gpu_scal(count, loss_weight, diff_.mutable_gpu_data());

    // mapping the gradient of diff_ (N, C, num_objs) back to prediction (N, C, H, W)
    KeypointBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diff_.gpu_data(), mask, iloc, bottom_diff,
        channels, num_objs, prob_spatial_dim);
  }
  // for (int i = 0; i < 2; ++i) {
  //   if (propagate_down[i]) {
  //     const Dtype sign = (i == 0) ? 1 : -1;
  //     // const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
  //     const Dtype alpha = sign * top[0]->cpu_diff()[0] / (num_pos + Dtype(1e-4));
  //     caffe_gpu_axpby(
  //         count,                           // count
  //         alpha,                           // alpha
  //         diff_.gpu_data(),                // x
  //         Dtype(0),                        // beta
  //         bottom[i]->mutable_gpu_diff());  // y
  //     if (has_weights_) {
  //       // Scale by "inside" weight
  //       caffe_gpu_mul(
  //           count,
  //           bottom[2]->gpu_data(),
  //           bottom[i]->gpu_diff(),
  //           bottom[i]->mutable_gpu_diff());
  //       // Scale by "outside" weight
  //       caffe_gpu_mul(
  //           count,
  //           bottom[3]->gpu_data(),
  //           bottom[i]->gpu_diff(),
  //           bottom[i]->mutable_gpu_diff());
  //     }
  //   }
  // }
}

INSTANTIATE_LAYER_GPU_FUNCS(KP2SmoothL1LossLayer);

}  // namespace caffe
