#include <vector>

#include "caffe/layers/kp2_smooth_l1_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void KP2SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  sigma2_ = loss_param.sigma() * loss_param.sigma();
  has_weights_ = (bottom.size() >= 5);
  if (has_weights_) {
    CHECK_EQ(bottom.size(), 6) << "If weights are used, must specify both "
      "inside and outside weights";
  }
}

template <typename Dtype>
void KP2SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  // bottom 0: prediction with shape (N, C, H, W).
  // bottom 1: groundtruth with shape (N, C, max_objs) or (N, C, max_objs, 1).
  // bottom 2: mask with shape (N, max_objs) or (N, max_objs, 1, 1), 
  //           binary indicator of objects existence (1 or 0).
  // bottom 3: iloc with shape (N, max_objs) or (N, max_objs, 1, 1), 
  //           object spatial index in bottom_data (H,W) plane that calculated by (hW+w).
  // bottom 4: inside weights has same shape with groundtruth (N, C, max_objs) or (N, C, max_objs, 1).
  // bottom 5: outside weights has same shape with groundtruth (N, C, max_objs) or (N, C, max_objs, 1).
  // batch size
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[0]->num(), bottom[3]->num());
  // channel
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  // max_objs
  CHECK_EQ(bottom[1]->count(2), bottom[2]->count(1));
  CHECK_EQ(bottom[1]->count(2), bottom[3]->count(1));
  if (has_weights_) {
    // each object has a weight.
    CHECK_EQ(bottom[1]->num(), bottom[4]->num());
    CHECK_EQ(bottom[1]->num(), bottom[5]->num());
    CHECK_EQ(bottom[1]->channels(), bottom[4]->channels());
    CHECK_EQ(bottom[1]->channels(), bottom[5]->channels());
    CHECK_EQ(bottom[1]->count(2), bottom[4]->count(2));
    CHECK_EQ(bottom[1]->count(2), bottom[5]->count(2));
  }
  
  diff_.Reshape(bottom[1]->num(), bottom[1]->channels(),
    bottom[1]->height(), bottom[1]->width());
  errors_.Reshape(bottom[1]->num(), bottom[1]->channels(),
    bottom[1]->height(), bottom[1]->width());
}

template <typename Dtype>
void KP2SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void KP2SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(KP2SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(KP2SmoothL1LossLayer);
REGISTER_LAYER_CLASS(KP2SmoothL1Loss);

}  // namespace caffe
