/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIVO_POINT_H_
#define LIVO_POINT_H_

#include <boost/noncopyable.hpp>
#include "common_lib.h"
#include "frame.h"

class Feature;

/**
 * @brief 场景表面的视觉地图点类
 * 用于表示和管理SLAM系统中的视觉特征点
 * 继承自boost::noncopyable以防止对象拷贝
 */
class VisualPoint : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 确保类的内存对齐

  Vector3d pos_;                // 点在世界坐标系下的3D位置
  Vector3d normal_;             // 点所在表面的法向量
  Matrix3d normal_information_; // 法向量估计的信息矩阵(协方差矩阵的逆)
  Vector3d previous_normal_;    // 上一次更新的法向量
  list<Feature *> obs_;         // 观测到该点的特征块列表
  Eigen::Matrix3d covariance_;  // 点位置的协方差矩阵
  bool is_converged_;           // 点是否已收敛的标志
  bool is_normal_initialized_;  // 法向量是否已初始化的标志
  bool has_ref_patch_;          // 是否有参考图像块的标志
  Feature *ref_patch;           // 参考图像块指针

  /**
   * @brief 构造函数
   * @param pos 点的初始3D位置
   */
  VisualPoint(const Vector3d &pos);

  /**
   * @brief 析构函数
   * 负责清理观测列表中的特征对象
   */
  ~VisualPoint();

  /**
   * @brief 查找得分最低的特征
   * @param framepos 当前帧位置
   * @param ftr 输出找到的特征指针
   */
  void findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const;

  /**
   * @brief 删除非参考图像块的特征
   */
  void deleteNonRefPatchFeatures();

  /**
   * @brief 删除指定特征的引用
   * @param ftr 要删除的特征指针
   */
  void deleteFeatureRef(Feature *ftr);

  /**
   * @brief 添加新的帧引用
   * @param ftr 要添加的特征指针
   */
  void addFrameRef(Feature *ftr);

  /**
   * @brief 获取最接近的视角观测
   * @param pos 当前位置
   * @param obs 输出找到的观测特征
   * @param cur_px 当前像素坐标
   * @return 是否找到合适的观测
   */
  bool getCloseViewObs(const Vector3d &pos, Feature *&obs, const Vector2d &cur_px) const;
};

#endif // LIVO_POINT_H_
