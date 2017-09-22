#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/roi_coordinate_layer.hpp"
using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void ROICoordinateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROICoordinateParameter roi_coordinate_param = this->layer_param_.roi_coordinate_param();
  pad_ = roi_coordinate_param.pad();
  threshold_ = roi_coordinate_param.threshold();
  CHECK_GE(pad_, 0)
      << "pad must be >= 0";
  CHECK_GE(threshold_, 0)
      << "threshold must be >= 0";
}

template <typename Dtype>
void ROICoordinateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // mask shape should be (n,1,h,w)
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  /* ROI
  *   top[1] shape should be (R, 5) or (R, 5, 1, 1)
  *   [R x 5] containing a list R ROI tuples with batch index and coordinates of
  *   regions of interest. Each row in top[1] is a ROI tuple in format
  *   [batch_index x1 y1 x2 y2], where batch_index corresponds to the index of
  *   instance in the first input and x1 y1 x2 y2 are 0-indexed coordinates
  *   of ROI rectangle (including its boundaries).
  */
  top[1]->Reshape(bottom[0]->num(), 5, 1, 1);
  // Init
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
}

template <typename Dtype>
void ROICoordinateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_rois = top[1]->mutable_cpu_data();
  // Init
  caffe_set(top[0]->count(), Dtype(0), top_data);
  caffe_set(top[1]->count(), Dtype(0), top_rois);
  
  int dim = bottom[0]->shape(1);
  // Distance between values of axis in blob
  int axis_dist = bottom[0]->count(1) / dim;
  int num = bottom[0]->count() / dim;
  int top_k = 1;

  std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector[j] = std::make_pair(
        bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    for (int j = 0; j < top_k; ++j) {
      // Produces max_ind per axis
      // top_data[(i / axis_dist * top_k + j) * axis_dist + i % axis_dist]
          // = bottom_data_vector[j].second;
      // top_data[(i / axis_dist * top_k + j) * axis_dist + i % axis_dist]
      //     = bottom_data_vector[j].second > 0 ? 1:0;
      // extract foreground
      if ((bottom_data_vector[j].first >= threshold_) && (bottom_data_vector[j].second > 0)) {
        top_data[(i / axis_dist * top_k + j) * axis_dist + i % axis_dist] = 1;
      }
    }
  }

  int batch_size = top[0]->num();
  int height = top[0]->height();
  int width = top[0]->width();
  for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
    int x1, y1, x2, y2;
    x1 = y1 = INT_MAX;
    x2 = y2 = INT_MIN;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        if (top_data[(batch_ind * height + h) * width + w] > 0) {
          x1 = (x1 < w) ? x1 : w;
          y1 = (y1 < h) ? y1 : h;
          x2 = (x2 > w) ? x2 : w;
          y2 = (y2 > h) ? y2 : h;
        }
      }
    }
    if (x1 > x2) {
      x1 = 0;
      x2 = width - 1;
    }
    if (y1 > y2) {
      y1 = 0;
      y2 = height - 1;
    }
    //convert rectangle to square
    int rectangle_width = x2 - x1 + 1;
    int rectangle_height = y2 - y1 + 1;
    int square_length = max(rectangle_width, rectangle_height) + 2 * pad_;
    // square_length can not greater than height and width
    square_length = (square_length < height) ? square_length : height;
    square_length = (square_length < width) ? square_length: width; 
    // update x1
    if (rectangle_width < square_length) {
      x1 -= (square_length - rectangle_width)/2;
    }
    // update y1
    if (rectangle_height < square_length) {
      y1 -= (square_length - rectangle_height)/2;
    }
    // prevent x1, y1 out of boundary
    x1 = max(x1, 0);
    y1 = max(y1, 0);
    x2 = square_length + x1 - 1;
    y2 = square_length + y1 - 1;
    // prevent x2, y2 out of boundary
    if (x2 > (width - 1)) {
      x2 = width - 1;
      x1 = x2 - square_length + 1;
    }
    if (y2 > (height - 1)) {
      y2 = height - 1;
      y1 = y2 - square_length + 1;
    }
    // x1 = max(x1 - pad_, 0);
    // y1 = max(y1 - pad_, 0);
    // x2 = min(x2 + pad_, width - 1);
    // y2 = min(y2 + pad_, height - 1);

    top_rois[0] = batch_ind;
    top_rois[1] = x1;
    top_rois[2] = y1;
    top_rois[3] = x2;
    top_rois[4] = y2;
    top_rois += 5;
  }
}

template <typename Dtype>
void ROICoordinateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(ROICoordinateLayer);
REGISTER_LAYER_CLASS(ROICoordinate);

}  // namespace caffe
