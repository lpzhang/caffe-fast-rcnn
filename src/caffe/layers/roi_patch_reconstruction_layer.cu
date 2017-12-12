#include <vector>

#include "caffe/layers/roi_patch_reconstruction_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ROIPatchReconstructionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int i = 0; i < num_; ++i) {
    // init in_ and out_
    caffe_gpu_set(in_.count(), Dtype(0), in_.mutable_gpu_data());
    caffe_gpu_set(out_.count(), Dtype(0), out_.mutable_gpu_data());
    // copy one batch bottom_data to in_
    caffe_copy(in_.count(), bottom_data, in_.mutable_gpu_data());
    // obtain ROI
    // int roi_level = static_cast<int>(bottom_rois[0]);
    int x1 = static_cast<int>(bottom_rois[1]);
    int y1 = static_cast<int>(bottom_rois[2]);
    int roi_width = static_cast<int>(bottom_rois[3]) - x1 + 1;
    int roi_height = static_cast<int>(bottom_rois[4]) - y1 + 1;
    LOG(INFO) << x1 << y1 << roi_width << roi_height;
    LOG(INFO) << height_in_ << width_in_ << height_out_ << width_out_;
    caffe_gpu_interp2<Dtype, false>(1 * channels_,
      in_.gpu_data(), 0, 0,
      height_in_, width_in_, height_in_, width_in_,
      out_.mutable_gpu_data(), x1, y1,
      roi_height, roi_width, height_out_, width_out_);
    // copy out_ to top_data
    caffe_copy(out_.count(), out_.gpu_data(), top_data);
    // next batch
    bottom_data += bottom[0]->count(1);
    bottom_rois += bottom[1]->count(1);
    top_data += top[0]->count(1);
  }
  // caffe_gpu_interp2<Dtype, false>(num_ * channels_,
  //   bottom[0]->gpu_data(), -pad_beg_, -pad_beg_,
  //   height_in_eff_, width_in_eff_, height_in_, width_in_,
  //   top[0]->mutable_gpu_data(), 0, 0,
  //   height_out_, width_out_, height_out_, width_out_);
}

template <typename Dtype>
void ROIPatchReconstructionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  for (int i = 0; i < num_; ++i) {
    // init in_ and out_
    caffe_gpu_set(in_.count(), Dtype(0), in_.mutable_gpu_data());
    caffe_gpu_set(out_.count(), Dtype(0), out_.mutable_gpu_data());
    // copy one batch top_diff to out_
    caffe_copy(out_.count(), top_diff, out_.mutable_gpu_data());
    // obtain ROI
    // int roi_level = static_cast<int>(bottom_rois[0]);
    int x1 = static_cast<int>(bottom_rois[1]);
    int y1 = static_cast<int>(bottom_rois[2]);
    int roi_width = static_cast<int>(bottom_rois[3]) - x1 + 1;
    int roi_height = static_cast<int>(bottom_rois[4]) - y1 + 1;
    caffe_gpu_interp2_backward<Dtype, false>(1 * channels_,
      in_.mutable_gpu_data(), 0, 0,
      height_in_, width_in_, height_in_, width_in_,
      out_.gpu_data(), x1, y1,
      roi_height, roi_width, height_out_, width_out_);
    // copy in_ to bottom_diff
    caffe_copy(in_.count(), in_.gpu_data(), bottom_diff);

    bottom_rois += bottom[1]->count(1);
    top_diff += top[0]->count(1);
    bottom_diff += bottom[0]->count(1);
  }
  // caffe_gpu_interp2_backward<Dtype, false>(num_ * channels_,
  //   bottom[0]->mutable_gpu_diff(), -pad_beg_, -pad_beg_,
  //   height_in_eff_, width_in_eff_, height_in_, width_in_,
  //   top[0]->gpu_diff(), 0, 0,
  //   height_out_, width_out_, height_out_, width_out_);
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIPatchReconstructionLayer);

}  // namespace caffe
