/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS()
  {
    for (VisualPoint *vp : voxel_points)
    {
      if (vp != nullptr)
      {
        delete vp;
        vp = nullptr;
      }
    }
  }
};

class VIOManager
{
public:
  // 图像网格大小
  int grid_size;
  // 相机模型指针
  vk::AbstractCamera *cam;
  // 针孔相机模型指针
  vk::PinholeCamera *pinhole_cam;
  // 当前状态指针
  StatesGroup *state;
  // 预测状态指针
  StatesGroup *state_propagat;
  // 旋转矩阵和雅可比矩阵:
  // Rli: IMU到LiDAR的旋转矩阵
  // Rci: 相机到IMU的旋转矩阵
  // Rcl: 相机到LiDAR的旋转矩阵
  // Rcw: 相机到世界坐标系的旋转矩阵
  // Jdphi_dR, Jdp_dt, Jdp_dR: 雅可比矩阵
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  // 平移向量:
  // Pli: IMU到LiDAR的平移
  // Pci: 相机到IMU的平移
  // Pcl: 相机到LiDAR的平移
  // Pcw: 相机到世界坐标系的平移
  V3D Pli, Pci, Pcl, Pcw;
  // 网格类型标记
  vector<int> grid_num;
  // 地图索引
  vector<int> map_index;
  // 边界标记
  vector<int> border_flag;
  // 更新标记
  vector<int> update_flag;
  // 地图距离
  vector<float> map_dist;
  // 扫描值
  vector<float> scan_value;
  // 特征块缓存
  vector<float> patch_buffer;
  // 功能开关标志:
  // normal_en: 法向量估计使能
  // inverse_composition_en: 逆向组合法使能
  // exposure_estimate_en: 曝光估计使能
  // raycast_en: 光线投射使能
  // has_ref_patch_cache: 是否有参考特征块缓存
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  // ncc_en: 归一化互相关使能
  // colmap_output_en: COLMAP输出使能
  bool ncc_en = false, colmap_output_en = false;

  // 图像和网格参数
  int width, height;               // 图像宽高
  int grid_n_width, grid_n_height; // 网格数量
  int length;                      // 总网格数
  double image_resize_factor;      // 图像缩放因子
  double fx, fy, cx, cy;           // 相机内参

  // 特征块参数
  int patch_pyrimid_level; // 金字塔层数
  int patch_size;          // 特征块大小
  int patch_size_total;    // 特征块总像素数
  int patch_size_half;     // 特征块半宽
  int border;              // 边界大小
  int warp_len;            // 变形长度
  int max_iterations;      // 最大迭代次数
  int total_points;        // 总点数

  // 阈值参数
  double img_point_cov;     // 图像点协方差
  double outlier_threshold; // 外点阈值
  double ncc_thre;          // 归一化互相关阈值

  // 子地图和光线采样
  SubSparseMap *visual_submap;                           // 视觉子地图
  std::vector<std::vector<V3D>> rays_with_sample_points; // 带采样点的射线

  // 时间统计
  double compute_jacobian_time; // 计算雅可比时间
  double update_ekf_time;       // 更新EKF时间
  double ave_total = 0;         // 平均总时间
  // double ave_build_residual_time = 0;  // 平均构建残差时间
  // double ave_ekf_time = 0;             // 平均EKF时间

  // 帧计数和绘图标志
  int frame_count = 0; // 帧计数器
  bool plot_flag;      // 绘图标志

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map;
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref,
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);

  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_