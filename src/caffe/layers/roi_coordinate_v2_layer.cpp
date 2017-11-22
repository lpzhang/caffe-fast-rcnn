#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/roi_coordinate_v2_layer.hpp"
using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void ROICoordinateV2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROICoordinateV2Parameter roi_coordinate_v2_param = this->layer_param_.roi_coordinate_v2_param();
  roi_num_ = roi_coordinate_v2_param.roi_num();
  threshold_ = roi_coordinate_v2_param.threshold();
  int point_size = roi_coordinate_v2_param.point_size();
  int scale_size = roi_coordinate_v2_param.scale_size();
  CHECK_GE(point_size, 0) << "point_size must be >= 0";
  CHECK_GE(scale_size, 0) << "scale_size must be >= 0";
  CHECK_GE(roi_num_, 0) << "roi_num must be >= 0";
  CHECK_EQ(roi_num_, scale_size - point_size) << "scale_size - point_size must equal to roi_num";
  LOG(INFO) << roi_num_;
  // Reshape 
  point_.Reshape(point_size, 1, 1, 1);
  scale_.Reshape(scale_size, 1, 1, 1);
  Dtype* point_data = point_.mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  for (int i = 0; i < point_size; ++i) {
    point_data[i] = roi_coordinate_v2_param.point(i);
  }
  for (int i = 0; i < scale_size; ++i) {
    scale_data[i] = roi_coordinate_v2_param.scale(i);
  }
}

template <typename Dtype>
void ROICoordinateV2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // mask shape should be (n, 1, h, w)
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  /* ROI
  *   top[1] shape should be (R, 5) or (R, 5, 1, 1)
  *   [R x 5] containing a list R ROI tuples with batch index and coordinates of
  *   regions of interest. Each row in top[1] is a ROI tuple in format
  *   [batch_index x1 y1 x2 y2], where batch_index corresponds to the index of
  *   instance in the first input and x1 y1 x2 y2 are 0-indexed coordinates
  *   of ROI rectangle (including its boundaries).
  */
  int R = bottom[0]->num() * roi_num_;
  top[1]->Reshape(R, 5, 1, 1);
  // Init
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
}

template <typename Dtype>
void ROICoordinateV2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* point_data = point_.cpu_data();
  const Dtype* scale_data = scale_.cpu_data();
  int point_size = point_.num();
  int scale_size = scale_.num();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_rois = top[1]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  caffe_set(top[1]->count(), Dtype(0), top_rois);

  // Produce Mask
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

  // Produce ROI
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
    // Get Original ROI Width and Height
    int rectangle_width = x2 - x1 + 1;
    int rectangle_height = y2 - y1 + 1;
    //convert rectangle to square
    int square_length = max(rectangle_width, rectangle_height);
    /* Look for the threshold interval that square_length belongs to
      and its corresponding scale value for each roi
    */
    int point_dist = point_size / roi_num_;
    int scale_dist = scale_size / roi_num_;
    for (int roi_ind = 0; roi_ind < roi_num_; ++roi_ind) {
      int point_index = roi_ind * point_dist;
      int scale_index = roi_ind * scale_dist;
      for (int p = 0; p < point_dist; ++p) {
        if (square_length >= point_data[point_index + p]) {
          scale_index += 1;
        } else {
          break;
        }
      }
      Dtype scale_value = scale_data[scale_index];
      // roi_length, roi_x1, roi_y1, roi_x2, roi_y2
      int roi_length = static_cast<int>(height / scale_value);
      int roi_x1 = x1;
      int roi_y1 = y1;
      int roi_x2 = x2;
      int roi_y2 = y2;
      // update roi_x1
      if (rectangle_width < roi_length) {
        roi_x1 -= (roi_length - rectangle_width)/2;
      }
      // update roi_y1
      if (rectangle_height < roi_length) {
        roi_y1 -= (roi_length - rectangle_height)/2;
      }
      // prevent roi_x1, roi_y1 out of boundary
      roi_x1 = max(roi_x1, 0);
      roi_y1 = max(roi_y1, 0);
      roi_x2 = roi_length + roi_x1 - 1;
      roi_y2 = roi_length + roi_y1 - 1;
      // prevent roi_x2, roi_y2 out of boundary
      if (roi_x2 > (width - 1)) {
        roi_x2 = width - 1;
        roi_x1 = roi_x2 - roi_length + 1;
      }
      if (roi_y2 > (height - 1)) {
        roi_y2 = height - 1;
        roi_y1 = roi_y2 - roi_length + 1;
      }
      top_rois[0] = batch_ind;
      top_rois[1] = roi_x1;
      top_rois[2] = roi_y1;
      top_rois[3] = roi_x2;
      top_rois[4] = roi_y2;
      top_rois += 5;
    }
  }
}

template <typename Dtype>
void ROICoordinateV2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(ROICoordinateV2Layer);
REGISTER_LAYER_CLASS(ROICoordinateV2);

}  // namespace caffe
