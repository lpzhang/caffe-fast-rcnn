#include <vector>

#include "caffe/layers/roi_patch_reconstruction_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ROIPatchReconstructionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ROIPatchReconstructionParameter roi_patch_reconstruction_param = this->layer_param_.roi_patch_reconstruction_param();
  CHECK_GE(roi_patch_reconstruction_param.height(), 0)
      << "output height must be >= 0";
  CHECK_GE(roi_patch_reconstruction_param.width(), 0)
      << "output width must be >= 0";

  height_out_ = roi_patch_reconstruction_param.height();
  width_out_ = roi_patch_reconstruction_param.width();
  // int num_specs = 0;
  // num_specs += interp_param.has_zoom_factor();
  // num_specs += interp_param.has_shrink_factor();
  // num_specs += interp_param.has_height() && interp_param.has_width();
  // CHECK_EQ(num_specs, 1) << "Output dimension specified either by "
  //     << "zoom factor or shrink factor or explicitly";
  // pad_beg_ = interp_param.pad_beg();
  // pad_end_ = interp_param.pad_end();
  // CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
  // CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Dtype>
void ROIPatchReconstructionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  CHECK_EQ(num_, bottom[1]->num()) << "Num of Batch must equal to num of ROIs";
  CHECK_GT(height_in_, 0) << "height of input should be positive";
  CHECK_GT(width_in_, 0) << "width of input should be positive";
  top[0]->Reshape(num_, channels_, height_out_, width_out_);
  in_.Reshape(1, channels_, height_in_, width_in_);
  out_.Reshape(1, channels_, height_out_, width_out_);

  // Init top[0] to 0 and Set background(channel 0) have prob 1.0
  // caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  // Dtype* top_data = top[0]->mutable_cpu_data();
  // for (int i = 0; i < num_; ++i) {
  //   caffe_set(top[0]->count(2), Dtype(1), top_data + top[0]->offset(i));
  // }
  // caffe_set(in_.count(), Dtype(0), in_.mutable_cpu_data());
  caffe_set(out_.count(), Dtype(0), out_.mutable_cpu_data());
  caffe_set(out_.count(2), Dtype(1), out_.mutable_cpu_data());
}

template <typename Dtype>
void ROIPatchReconstructionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // Init top[0] to 0 and Set background(channel 0) have prob 1.0
  // caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  // for (int i = 0; i < num_; ++i) {
  //   caffe_set(top[0]->count(2), Dtype(1), top_data + top[0]->offset(i));
  // }
  caffe_set(out_.count(), Dtype(0), out_.mutable_cpu_data());
  caffe_set(out_.count(2), Dtype(1), out_.mutable_cpu_data());
  for (int i = 0; i < num_; ++i) {
    // copy 1 batch data to in_
    caffe_copy(in_.count(), bottom_data, in_.mutable_cpu_data());
    // obtain ROI
    // int roi_level = static_cast<int>(bottom_rois[0]);
    int x1 = static_cast<int>(bottom_rois[1]);
    int y1 = static_cast<int>(bottom_rois[2]);
    int roi_width = static_cast<int>(bottom_rois[3]) - x1 + 1;
    int roi_height = static_cast<int>(bottom_rois[4]) - y1 + 1;
    // LOG(INFO) << x1 << y1 << roi_width << roi_height;
    // LOG(INFO) << height_in_ << width_in_ << height_out_ << width_out_;
    caffe_cpu_interp2<Dtype, false>(1 * channels_,
      in_.cpu_data(), 0, 0,
      height_in_, width_in_, height_in_, width_in_,
      out_.mutable_cpu_data(), x1, y1,
      roi_height, roi_width, height_out_, width_out_);
    // copy out_ to top_data
    caffe_copy(out_.count(), out_.cpu_data(), top_data);

    bottom_data += bottom[0]->count(1);
    bottom_rois += bottom[1]->count(1);
    top_data += top[0]->count(1);
  }
  
  // int roi_level = int(bottom_rois[0]);
  // int x1 = static_cast<int>(bottom_rois[1]);
  // int y1 = static_cast<int>(bottom_rois[2]);
  // int roi_width = static_cast<int>(bottom_rois[3]) - x1 + 1;
  // int roi_height = static_cast<int>(bottom_rois[4]) - y1 + 1;
  
  // caffe_cpu_interp2<Dtype, false>(num_ * channels_,
  //   bottom[0]->cpu_data(), 0, 0,
  //   height_in_, width_in_, height_in_, width_in_,
  //   top[0]->mutable_cpu_data(), x1, y1,
  //   roi_height, roi_width, height_out_, width_out_);

  // caffe_cpu_interp2<Dtype, false>(num_ * channels_,
  //     bottom[0]->cpu_data(), -pad_beg_, -pad_beg_,
  //     height_in_eff_, width_in_eff_, height_in_, width_in_,
  //     top[0]->mutable_cpu_data(), 0, 0,
  //     height_out_, width_out_, height_out_, width_out_);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ROIPatchReconstructionLayer, Forward);
#endif

INSTANTIATE_CLASS(ROIPatchReconstructionLayer);
REGISTER_LAYER_CLASS(ROIPatchReconstruction);

}  // namespace caffe
