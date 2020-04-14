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
__global__ void KeypointForwardGPU(const int nthreads,
    const Dtype* bottom_data, const Dtype* bottom_label, const Dtype* bottom_mask, const Dtype* bottom_iloc,
    const int channels, const int max_objs, const int spatial_dim,
    Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, m) is an element index in bottom_label (N, C, max_objs) or (N, C, max_objs, 1)
    const int m = index % max_objs;
    const int c = index / max_objs % channels;
    const int n = index / max_objs / channels;

    // bottom_mask with shape (N, max_objs) or (N, max_objs, 1, 1),
    // index mapping: bottom_label (n, c, m) => (n, m) bottom_mask
    const int indicator = static_cast<int>(bottom_mask[n * max_objs + m]);
    // get stored object spatial index in bottom_data (H,W) plane that calculated by (hW+w). 
    const int spatial_ind = static_cast<int>(bottom_iloc[n * max_objs + m]);
    // the corresponding index mapping: bottom_label (n, c, m) => bottom_data (n, c, h, w)
    // here, the (h, w) index is represented by spatial_ind.
    const int bottom_data_ind = n * channels * spatial_dim + c * spatial_dim + spatial_ind;

    if (indicator > 0) {
      diff[index] = bottom_data[bottom_data_ind] - bottom_label[index];
    }
  }
}

template <typename Dtype>
void KP2SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // prediction with shape (N, C, H, W), 
  // groundtruth with shape (N, C, max_objs) or (N, C, max_objs, 1),
  // mask (N, max_objs) or (N, max_objs, 1, 1)
  // iloc (N, max_objs) or (N, max_objs, 1, 1)
  const Dtype* bottom_data  = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* bottom_mask  = bottom[2]->gpu_data();
  const Dtype* bottom_iloc  = bottom[3]->gpu_data();
  const int nthreads = bottom[1]->count();
  const int channels = bottom[1]->channels();
  const int max_objs = bottom[1]->count(2);
  // spatial_dim (H*W) of the bottom_data
  const int spatial_dim = bottom[0]->count(2);
  // Compute the keypoint diff_ w_in * (b0 - b1)
  // NOLINT_NEXT_LINE(whitespace/operators)
  KeypointForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads,
      bottom_data, bottom_label, bottom_mask, bottom_iloc, channels, max_objs, spatial_dim,
      diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  // caffe_gpu_sub(
  //     count,
  //     bottom[0]->gpu_data(),
  //     bottom[1]->gpu_data(),
  //     diff_.mutable_gpu_data());    // d := b0 - b1
  // inside weights bottom[4]->gpu_data() (N, C, max_objs) or (N, C, max_objs, 1)
  if (has_weights_) {
    // apply "inside" weights
    caffe_gpu_mul(
        nthreads,
        bottom[4]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w_in * (b0 - b1)
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;
  // outside weights bottom[5]->gpu_data() (N, C, num_objs)
  if (has_weights_) {
    // apply "outside" weights
    caffe_gpu_mul(
        nthreads,
        bottom[5]->gpu_data(),
        errors_.gpu_data(),
        errors_.mutable_gpu_data());  // d := w_out * SmoothL1(w_in * (b0 - b1))
  }

  Dtype loss;
  caffe_gpu_asum(nthreads, errors_.gpu_data(), &loss);
  // the number of positive objects by summing mask (N, max_objs) or (N, max_objs, 1, 1)
  Dtype num_pos = 0;
  caffe_gpu_asum(bottom[2]->count(), bottom_mask, &num_pos);
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
__global__ void KeypointBackwardGPU(const int nthreads,
    const Dtype* diff, const Dtype* bottom_mask, const Dtype* bottom_iloc,
    const int channels, const int max_objs, const int spatial_dim,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, m) is an element index in bottom_label (N, C, max_objs) or (N, C, max_objs, 1)
    const int m = index % max_objs;
    const int c = index / max_objs % channels;
    const int n = index / max_objs / channels;

    // bottom_mask with shape (N, max_objs) or (N, max_objs, 1, 1),
    // index mapping: bottom_label (n, c, m) => (n, m) bottom_mask
    const int indicator = static_cast<int>(bottom_mask[n * max_objs + m]);
    // get stored object spatial index in bottom_data (H,W) plane that calculated by (hW+w). 
    const int spatial_ind = static_cast<int>(bottom_iloc[n * max_objs + m]);
    // the corresponding index mapping: bottom_label (n, c, m) => bottom_data (n, c, h, w)
    // here, the (h, w) index is represented by spatial_ind.
    const int bottom_data_ind = n * channels * spatial_dim + c * spatial_dim + spatial_ind;

    if (indicator > 0) {
      bottom_diff[bottom_data_ind] = diff[index];
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
    Dtype* bottom_diff       = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_mask = bottom[2]->gpu_data();
    const Dtype* bottom_iloc = bottom[3]->gpu_data();
    const int nthreads = bottom[1]->count();
    const int channels = bottom[1]->channels();
    const int max_objs = bottom[1]->count(2);
    // spatial_dim (H*W) of the bottom_data
    const int spatial_dim = bottom[0]->count(2);
    // after forwards, diff_ holds w_in * (b0 - b1)
    // NOLINT_NEXT_LINE(whitespace/operators)
    SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, diff_.gpu_data(), diff_.mutable_gpu_data(), sigma2_);
    CUDA_POST_KERNEL_CHECK;
    
    // Scale diff_ before mapping back to bottom_diff, since diff_ and weights have same dimensions.
    if (has_weights_) {
      // Scale by "inside" weight bottom[4]
      caffe_gpu_mul(nthreads, bottom[4]->gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
      // Scale by "outside" weight bottom[5]
      caffe_gpu_mul(nthreads, bottom[5]->gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
    }
    // Scale gradient
    // the number of positive objects by summing mask (N, max_objs) or (N, max_objs, 1, 1)
    Dtype num_pos = 0;
    caffe_gpu_asum(bottom[2]->count(), bottom_mask, &num_pos);
    const Dtype loss_weight = top[0]->cpu_diff()[0] / (num_pos + Dtype(1e-4));
    caffe_gpu_scal(nthreads, loss_weight, diff_.mutable_gpu_data());

    // mapping the gradient of diff_ (N, C, max_objs, 1) back to bottom_diff (N, C, H, W)
    KeypointBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads,
        diff_.gpu_data(), bottom_mask, bottom_iloc, channels, max_objs, spatial_dim,
        bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KP2SmoothL1LossLayer);

}  // namespace caffe
