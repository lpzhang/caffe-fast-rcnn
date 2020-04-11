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
  // bottom0 prediction (N, C, H, W)
  // bottom1 groundtruth (N, C, num_objs)
  // bottom2 mask (N, num_objs)
  // bottom3 iloc (N, num_objs)
  // bottom4 inside weights (N, C, num_objs)
  // bottom5 outside weights (N, C, num_objs)
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "batch size should be the same.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->num())
      << "batch size should be the same.";
  CHECK_EQ(bottom[0]->num(), bottom[3]->num())
      << "batch size should be the same.";

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
      << "numbers of channels should be the same for prediction (n,c,h,w) and groundtruth (n,c,num_objs).";
  
  CHECK_EQ(bottom[1]->count(2), bottom[2]->count(1))
      << "groundtruth (n, c, num_objs) and mask (n, num_objs) should have the same num_objs.";
  CHECK_EQ(bottom[1]->count(2), bottom[3]->count(1))
      << "groundtruth (n, c, num_objs) and the spatial location index iloc(n, num_objs)"
      << "should have same num_objs.";

  if (has_weights_) {
    CHECK_EQ(bottom[1]->num(), bottom[4]->num())
        << "batch size should be the same.";
    CHECK_EQ(bottom[1]->num(), bottom[5]->num())
        << "batch size should be the same.";
    CHECK_EQ(bottom[1]->channels(), bottom[4]->channels())
        << "numbers of channels should be the same.";
    CHECK_EQ(bottom[1]->channels(), bottom[5]->channels())
        << "numbers of channels should be the same.";
    CHECK_EQ(bottom[1]->height(), bottom[4]->height())
        << "height should be the same.";
    CHECK_EQ(bottom[1]->height(), bottom[5]->height())
        << "height should be the same.";
    CHECK_EQ(bottom[1]->width(), bottom[4]->width())
        << "width should be the same.";
    CHECK_EQ(bottom[1]->width(), bottom[5]->width())
        << "width should be the same.";
  }
  
  // diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
  //     bottom[0]->height(), bottom[0]->width());
  diff_.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  // errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
  //     bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  // vector of ones used to sum
  // ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
  //     bottom[0]->height(), bottom[0]->width());
  ones_.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  for (int i = 0; i < bottom[1]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
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
