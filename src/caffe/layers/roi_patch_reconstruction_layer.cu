// #include <vector>

// #include "caffe/layers/roi_patch_reconstruction_layer.hpp"
// #include "caffe/util/interp.hpp"
// #include "caffe/util/math_functions.hpp"

// namespace caffe {

// template <typename Dtype>
// void ROIPatchReconstructionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {
//   const Dtype* bottom_rois = bottom[1]->gpu_data();
//   // int roi_level = int(bottom_rois[0]);
//   int x1 = int(bottom_rois[1]);
//   int y1 = int(bottom_rois[2]);
//   int x2 = int(bottom_rois[3]);
//   int y2 = int(bottom_rois[4]);
//   int roi_width = x2 - x1 + 1;
//   int roi_height = y2 - y1 + 1;
//   LOG(INFO) << x1 << y1 << roi_width << roi_height;
//   LOG(INFO) << height_in_ << width_in_ << height_out_ << width_out_;
//   caffe_gpu_interp2<Dtype, false>(num_ * channels_,
//     bottom[0]->gpu_data(), 0, 0,
//     height_in_, width_in_, height_in_, width_in_,
//     top[0]->mutable_gpu_data(), x1, y1,
//     roi_height, roi_width, height_out_, width_out_);

//   // caffe_gpu_interp2<Dtype, false>(num_ * channels_,
//   //   bottom[0]->gpu_data(), -pad_beg_, -pad_beg_,
//   //   height_in_eff_, width_in_eff_, height_in_, width_in_,
//   //   top[0]->mutable_gpu_data(), 0, 0,
//   //   height_out_, width_out_, height_out_, width_out_);
// }

// template <typename Dtype>
// void ROIPatchReconstructionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//   NOT_IMPLEMENTED;
//   // if (!propagate_down[0]) { return; }
//   // caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
//   // caffe_gpu_interp2_backward<Dtype, false>(num_ * channels_,
//   //   bottom[0]->mutable_gpu_diff(), -pad_beg_, -pad_beg_,
//   //   height_in_eff_, width_in_eff_, height_in_, width_in_,
//   //   top[0]->gpu_diff(), 0, 0,
//   //   height_out_, width_out_, height_out_, width_out_);
// }

// INSTANTIATE_LAYER_GPU_FUNCS(ROIPatchReconstructionLayer);

// }  // namespace caffe
