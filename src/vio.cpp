/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager()
{
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO()
{
  // 创建视觉子地图对象
  visual_submap = new SubSparseMap;

  // 从相机模型获取内参
  fx = cam->fx();  // 相机焦距x
  fy = cam->fy();  // 相机焦距y 
  cx = cam->cx();  // 光心x坐标
  cy = cam->cy();  // 光心y坐标
  image_resize_factor = cam->scale();  // 图像缩放因子

  // 打印相机内参
  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  // 获取图像宽高
  width = cam->width();
  height = cam->height();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  
  // 计算相机到IMU的外参
  Rci = Rcl * Rli;  // 旋转矩阵
  Pci = Rcl * Pli + Pcl;  // 平移向量

  // 计算雅可比矩阵
  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;

  // 计算图像网格大小
  if (grid_size > 10)
  {
    // 如果网格大小大于10,根据网格大小计算网格数量
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  }
  else
  {
    // 否则根据网格数量计算网格大小
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  length = grid_n_width * grid_n_height;  // 总网格数

  // 如果启用光线投射
  if(raycast_en)
  {
    // 初始化边界标志
    border_flag.resize(length, 0);

    // 初始化带采样点的射线
    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", 
           grid_size, grid_n_height, grid_n_width, length);

    // 设置采样参数
    float d_min = 0.1;  // 最小深度
    float d_max = 3.0;  // 最大深度
    float step = 0.2;   // 采样步长
    
    // 遍历每个网格
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        // 标记边界网格
        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) 
          border_flag[index] = 1;

        // 计算网格中心点坐标
        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        
        // 在每个深度生成采样点
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          xyz = cam->cam2world(u, v);  // 将像素坐标转换为世界坐标
          xyz *= d_temp / xyz[2];      // 缩放到指定深度
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
  }

  // 如果启用COLMAP输出
  if(colmap_output_en)
  {
    // 转换相机模型
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    
    // 打开并写入images.txt
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    
    // 打开并写入cameras.txt
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }

  // 初始化各种数组
  grid_num.resize(length);      // 网格类型
  map_index.resize(length);     // 地图索引
  map_dist.resize(length);      // 地图距离
  update_flag.resize(length);   // 更新标志
  scan_value.resize(length);    // 扫描值

  // 计算特征块相关参数
  patch_size_total = patch_size * patch_size;  // 特征块总像素数
  patch_size_half = static_cast<int>(patch_size / 2);  // 特征块半宽
  patch_buffer.resize(patch_size_total);  // 特征块缓冲区
  warp_len = patch_size_total * patch_pyrimid_level;  // 变形长度
  border = (patch_size_half + 2) * 8;  // 边界大小

  // 预分配体素点容器
  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  // 清空子特征地图
  sub_feat_map.clear();
}

// 重置网格相关数据的函数
void VIOManager::resetGrid()
{
  // 使用std::fill重置各个数组到初始状态
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);  // 将所有网格类型重置为未知
  fill(map_index.begin(), map_index.end(), 0);           // 将地图索引重置为0
  fill(map_dist.begin(), map_dist.end(), 10000.0f);      // 将地图距离重置为一个较大值10000.0
  fill(update_flag.begin(), update_flag.end(), 0);       // 将更新标志重置为0
  fill(scan_value.begin(), scan_value.end(), 0.0f);      // 将扫描值重置为0

  // 清空并重新分配体素点容器
  retrieve_voxel_points.clear();                         // 清空检索体素点容器
  retrieve_voxel_points.resize(length);                  // 重新分配大小为length的空间

  append_voxel_points.clear();                          // 清空追加体素点容器  
  append_voxel_points.resize(length);                   // 重新分配大小为length的空间

  total_points = 0;                                     // 重置总点数为0
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

// 计算投影雅可比矩阵的函数
// 输入:
// - p: 3D点坐标 (x,y,z)
// - J: 2x3的雅可比矩阵引用,用于存储计算结果
void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J) 
{
  // 提取3D点的x,y坐标
  const double x = p[0];
  const double y = p[1];
  
  // 计算z坐标的倒数及其平方
  const double z_inv = 1. / p[2];    // 1/z
  const double z_inv_2 = z_inv * z_inv;  // 1/z^2
  
  // 计算雅可比矩阵的各个元素
  // J = [fx/z     0    -fx*x/z^2]
  //     [0     fy/z    -fy*y/z^2]
  J(0, 0) = fx * z_inv;         // ∂u/∂x = fx/z
  J(0, 1) = 0.0;               // ∂u/∂y = 0
  J(0, 2) = -fx * x * z_inv_2;  // ∂u/∂z = -fx*x/z^2
  J(1, 0) = 0.0;               // ∂v/∂x = 0  
  J(1, 1) = fy * z_inv;         // ∂v/∂y = fy/z
  J(1, 2) = -fy * y * z_inv_2;  // ∂v/∂z = -fy*y/z^2
}

// 这个函数用于从图像中提取特征块(patch)
// 输入参数:
// - img: 输入图像
// - pc: 特征点在图像上的坐标(u,v)
// - patch_tmp: 用于存储提取的特征块的数组
// - level: 图像金字塔层级
void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  // 获取特征点的坐标
  const float u_ref = pc[0];  // x坐标
  const float v_ref = pc[1];  // y坐标
  
  // 计算当前层级的缩放因子,例如level=1时scale=2,level=2时scale=4
  const int scale = (1 << level);
  
  // 计算特征点所在的整数坐标
  const int u_ref_i = floorf(pc[0] / scale) * scale;  // x方向整数坐标
  const int v_ref_i = floorf(pc[1] / scale) * scale;  // y方向整数坐标
  
  // 计算亚像素偏移量
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;  // x方向偏移
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;  // y方向偏移
  
  // 计算双线性插值的四个权重
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);  // 左上角权重
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);          // 右上角权重  
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;          // 左下角权重
  const float w_ref_br = subpix_u_ref * subpix_v_ref;                  // 右下角权重

  // 遍历特征块的每个像素
  for (int x = 0; x < patch_size; x++)
  {
    // 计算图像中对应行的起始指针
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    
    // 遍历特征块每一行的像素
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      // 使用双线性插值计算特征块像素值
      // 将四个相邻像素按权重加权平均
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] +                    // 左上角像素
          w_ref_tr * img_ptr[scale] +                // 右上角像素
          w_ref_bl * img_ptr[scale * width] +        // 左下角像素
          w_ref_br * img_ptr[scale * width + scale]; // 右下角像素
    }
  }
}

// 这个函数用于将视觉特征点插入到体素地图中
// 输入参数:
// - pt_new: 待插入的视觉特征点指针
void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new) 
{
  // 获取特征点的世界坐标
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  
  // 设置体素大小为0.5米
  double voxel_size = 0.5;
  
  // 计算特征点所在的体素坐标
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    // 将世界坐标除以体素大小得到体素坐标
    loc_xyz[j] = pt_w[j] / voxel_size;
    // 对负坐标进行向下取整
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }

  // 创建体素位置对象
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  
  // 在体素地图中查找该位置
  auto iter = feat_map.find(position);
  
  if (iter != feat_map.end())
  {
    // 如果该体素已存在,将特征点添加到体素中并更新计数
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    // 如果该体素不存在,创建新的体素并添加特征点
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

// 这个函数用于计算基于单应性矩阵的仿射变换矩阵
// 输入参数:
// - cam: 相机模型
// - px_ref: 参考帧中的像素坐标
// - xyz_ref: 参考帧中的3D点坐标
// - normal_ref: 参考帧中的法向量
// - T_cur_ref: 当前帧到参考帧的变换矩阵
// - level_ref: 参考帧的金字塔层级
// - A_cur_ref: 输出的仿射变换矩阵
void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // 计算单应性矩阵
  // 1. 获取平移向量
  const V3D t = T_cur_ref.inverse().translation();
  
  // 2. 构建单应性矩阵 H = R(nI - tn^T)
  // 其中R是旋转矩阵,n是法向量,t是平移向量,I是单位矩阵
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());

  // 计算仿射变换矩阵
  const int kHalfPatchSize = 4;  // 特征块半宽度
  
  // 3. 计算参考帧中x和y方向的偏移向量
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));  // x方向
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));  // y方向

  // 4. 使用单应性矩阵投影参考点和偏移点到当前帧
  const V3D f_cur(H_cur_ref * xyz_ref);      // 投影参考点
  const V3D f_du_cur = H_cur_ref * f_du_ref; // 投影x方向偏移点
  const V3D f_dv_cur = H_cur_ref * f_dv_ref; // 投影y方向偏移点

  // 5. 将3D点投影到当前帧的图像平面
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));

  // 6. 计算仿射变换矩阵的两列
  // 第一列是x方向的变化率,第二列是y方向的变化率
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

// 这个函数用于将当前帧的特征块warp到参考帧
// 输入参数:
// - A_cur_ref: 当前帧到参考帧的仿射变换矩阵
// - img_ref: 参考帧的图像
// - px_ref: 参考帧中的像素坐标
// - level_ref: 参考帧的金字塔层级
// - search_level: 搜索的金字塔层级
// - pyramid_level: 金字塔层级
// - halfpatch_size: 特征块半宽度
// - patch: 输出特征块
void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  // 计算特征块大小
  const int patch_size = halfpatch_size * 2;

  // 计算从当前帧到参考帧的仿射变换矩阵,并转换为float类型
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();

  // 检查变换矩阵是否有效
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  // 遍历特征块的每个像素
  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x)
    {
      // 计算特征块中每个点相对于中心的偏移
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);

      // 根据金字塔层级缩放偏移量
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);

      // 使用仿射变换将点从当前帧变换到参考帧
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());

      // 检查变换后的点是否在图像范围内
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        // 如果超出范围,设置为0
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        // 如果在范围内,使用双线性插值获取像素值
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

// 这个函数用于确定最佳的图像金字塔搜索层级
// 输入参数:
// - A_cur_ref: 当前帧到参考帧的仿射变换矩阵
// - max_level: 最大允许的金字塔层级
// 返回值:
// - 最佳的搜索层级
int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // 初始化搜索层级为0(原始图像)
  int search_level = 0;
  
  // 计算仿射变换矩阵的行列式,表示变换引起的面积变化
  double D = A_cur_ref.determinant();
  
  // 当行列式大于3.0(表示图像块被放大了3倍以上)
  // 且当前层级小于最大层级时
  while (D > 3.0 && search_level < max_level)
  {
    // 增加搜索层级
    search_level += 1;
    // 每上升一层,面积变化缩小为原来的1/4
    D *= 0.25;
  }
  
  // 返回最佳搜索层级
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

// 这个函数用于从视觉稀疏地图中检索点
// 输入参数:
// - img: 输入图像
// - pg: 点云
// - plane_map: 平面地图
void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  // 如果特征地图为空则直接返回
  if (feat_map.size() <= 0) return;
  // 记录开始时间
  double ts0 = omp_get_wtime();

  // 重置视觉子地图
  visual_submap->reset();

  // 清空子特征地图
  sub_feat_map.clear();

  // 设置体素大小为0.5米
  float voxel_size = 0.5;

  // 如果不启用法向量,则清空变形地图
  if (!normal_en) warp_map.clear();

  // 创建深度图,初始化为0
  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // 用于存储体素坐标
  int loc_xyz[3];

  // 遍历点云中的每个点
  for (int i = 0; i < pg.size(); i++)
  {
    // 获取点的世界坐标
    V3D pt_w = pg[i].point_w;

    // 计算点所在的体素坐标
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      // 对负坐标进行向下取整
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    // 创建体素位置对象
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // 在子特征地图中查找或插入该位置
    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // 将点从世界坐标转换到相机坐标
    V3D pt_c(new_frame_->w2f(pt_w));

    // 如果点在相机前方
    if (pt_c[2] > 0)
    {
      V2D px;
      // 将点投影到图像平面
      px = new_frame_->cam_->world2cam(pt_c);

      // 如果点在图像边界内
      if (new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // 记录该点的深度到深度图中
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  // 遍历子特征地图中的每个位置
  for (auto &iter : sub_feat_map)
  {
    // 获取体素位置
    VOXEL_LOCATION position = iter.first;

    // 在特征地图中查找该位置
    auto corre_voxel = feat_map.find(position);

    // 如果该位置存在特征点
    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      // 获取该位置的所有特征点
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      // 获取特征点数量
      int voxel_num = voxel_points.size();

      // 遍历该位置的所有特征点
      for (int i = 0; i < voxel_num; i++)
      {
        // 获取特征点
        VisualPoint *pt = voxel_points[i];
        // 如果特征点为空则跳过
        if (pt == nullptr) continue;
        // 如果特征点没有观测则跳过
        if (pt->obs_.size() == 0) continue;

        // 获取特征点的法向量
        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        // 获取特征点的方向
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        // 如果特征点在相机后方则跳过
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        // 将特征点投影到图像平面
        V2D pc(new_frame_->w2c(pt->pos_));
        // 如果特征点在图像边界内
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // 将该位置标记为在视野内
          voxel_in_fov = true;
          // 计算特征点到相机中心的距离
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          // 如果距离小于当前最小距离,则更新最小距离和对应的特征点
          if (cur_dist <= map_dist[index])
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      // 如果该位置不在视野内,则将其从子特征地图中删除
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    // 遍历每个体素
    for (int i = 0; i < length; i++)
    {
      // 如果体素在地图中或边界标记为1,则跳过
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // 计算体素在图像中的行和列
      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      // 遍历该体素的所有射线
      for (const auto &it : rays_with_sample_points[i])
      {
        // 将射线转换到世界坐标
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          // 计算射线在世界坐标系中的体素坐标
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          // 如果体素坐标小于0,则向下取整
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }
        // 创建体素位置对象
        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
        // 在子特征地图中查找该位置
        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        // 如果该位置存在特征点,则跳过
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        // 在特征地图中查找该位置
        auto corre_feat_map = feat_map.find(sample_pos);
        // 如果该位置存在特征点,则跳过
        if (corre_feat_map != feat_map.end())
        {
          // 标记该位置是否在视野内
          bool voxel_in_fov = false;
          // 获取该位置的所有特征点
          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          // 获取特征点数量
          int voxel_num = voxel_points.size();
          // 如果特征点数量为0,则跳过
          if (voxel_num == 0) continue;
          // 遍历该位置的所有特征点
          for (int j = 0; j < voxel_num; j++)
          {
            // 获取特征点
            VisualPoint *pt = voxel_points[j];
            // 如果特征点为空则跳过
            if (pt == nullptr) continue;
            // 如果特征点没有观测则跳过
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            // 获取特征点的法向量
            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            // 获取特征点的方向
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            // 如果特征点在相机后方则跳过
            if (dir[2] < 0) continue;
            // 归一化方向向量
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            // 将特征点投影到图像平面
            V2D pc(new_frame_->w2c(pt->pos_));
            // 如果特征点在图像边界内
            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // 将该位置标记为在视野内
              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              // 计算特征点到相机中心的距离
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);
              // 计算距离
              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index])
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }
          // 如果该位置在视野内,则将该位置从子特征地图中删除
          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          // 创建体素位置对象
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          // 在平面地图中查找该位置
          auto iter = plane_map.find(sample_pos);
          // 如果该位置存在平面,则跳过
          if (iter != plane_map.end())
          {
            // 在八叉树中查找该位置
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            // 如果该位置存在平面,则跳过
            if (current_octo->plane_ptr_->is_plane_)
            {
              // 创建平面中心点对象
              pointWithVar plane_center;
              // 获取平面
              VoxelPlane &plane = *current_octo->plane_ptr_;
              // 将平面中心点赋值给平面中心点对象
              plane_center.point_w = plane.center_;
              // 将平面法向量赋值给平面中心点对象
              plane_center.normal = plane.normal_;
              // 将平面中心点对象添加到子特征地图中
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  // 遍历每个需要删除的体素位置
  for (auto &key : DeleteKeyList)
  {
    // 从子特征地图中删除该位置
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  // 遍历每个体素
  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      // 获取特征点
      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      // 将特征点投影到图像平面
      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      // 将特征点投影到相机坐标系
      V3D pt_cam(new_frame_->w2f(pt->pos_));
      // 创建深度连续标志
      bool depth_continous = false;
      // 遍历patch_size_half范围内的所有像素
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          // 如果像素为0,则跳过
          if (u == 0 && v == 0) continue;
          // 获取深度
          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];
          // 如果深度为0,则跳过 
          if (depth == 0.) continue;
          // 计算深度差
          double delta_dist = abs(pt_cam[2] - depth);
          // 如果深度差大于0.5,则将深度连续标志设置为true
          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      // 如果深度不连续,则跳过
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      // 创建特征点指针
      Feature *ref_ftr;
      // 创建patch_wrap
      std::vector<float> patch_wrap(warp_len);
      // 创建搜索等级
      int search_level;
      // 创建A_cur_ref_zero
      Matrix2d A_cur_ref_zero;
      // 如果特征点没有初始化则跳过
      if (!pt->is_normal_initialized_) continue;

      // 如果启用了法向量,则计算最小光度误差
      if (normal_en)
      {
        // 创建最小光度误差
        float phtometric_errors_min = std::numeric_limits<float>::max();
        // 如果特征点只有一个观测,则将该观测作为参考观测
        if (pt->obs_.size() == 1)
        {
          // 将该观测作为参考观测
          ref_ftr = *pt->obs_.begin();
          // 将参考观测赋值给特征点
          pt->ref_patch = ref_ftr;
          // 设置特征点有参考观测
          pt->has_ref_patch_ = true;
        }
        // 如果特征点没有参考观测,则计算最小光度误差
        else if (!pt->has_ref_patch_)
        {
          // 遍历特征点的所有观测
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            // 获取参考观测
            Feature *ref_patch_temp = *it;
            // 获取参考观测的patch
            float *patch_temp = ref_patch_temp->patch_;
            // 创建光度误差
            float phtometric_errors = 0.0;
            // 创建计数器
            int count = 0;
            // 遍历特征点的所有观测
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              // 如果参考观测和当前观测相同,则跳过
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              // 获取当前观测的patch
              float *patch_cache = (*itm)->patch_;
              // 遍历patch_size_total范围内的所有像素
              for (int ind = 0; ind < patch_size_total; ind++)
              {
                // 计算光度误差
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              // 增加计数器
              count++;
            }
            // 计算平均光度误差
            phtometric_errors = phtometric_errors / count;
            // 如果光度误差小于最小光度误差,则更新最小光度误差和参考观测
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          // 将参考观测赋值给特征点
          pt->ref_patch = ref_ftr;
          // 设置特征点有参考观测
          pt->has_ref_patch_ = true;
        }
        // 获取参考观测
        else { ref_ftr = pt->ref_patch; }
      }
      // 如果未启用法向量,则获取最近观测
      else
      {
        // 如果获取最近观测失败,则跳过
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en)
      {
        // 获取特征点的法向量
        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();
        // 获取特征点在世界坐标系中的位置
        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();
        
        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  if (total_points == 0) return;
  
  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

// 这个函数用于生成视觉地图点
void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  // 如果输入的点太少(<=10)就直接返回
  if (pg.size() <= 10) return;

  // 遍历所有输入的点
  for (int i = 0; i < pg.size(); i++)
  {
    // 跳过法向量为0的点
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    // 获取世界坐标系下的3D点位置
    V3D pt = pg[i].point_w;
    // 将3D点投影到相机平面上得到2D坐标
    V2D pc(new_frame_->w2c(pt));

    // 检查投影点是否在图像边界内
    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
    {
      // 计算点所在的网格索引
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      // 如果该网格不是地图点
      if (grid_num[index] != TYPE_MAP)
      {
        // 计算Shi-Tomasi角点响应值
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // 如果当前点的响应值更大,则更新该网格
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // 处理来自体素地图的点
  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // 创建新的视觉地图点
  int add = 0;
  for (int i = 0; i < length; i++)
  {
    // 处理标记为点云的网格
    if (grid_num[i] == TYPE_POINTCLOUD)
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;

      // 计算法向量和视线方向
      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      
      V2D pc(new_frame_->w2c(pt));

      // 提取图像块
      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      // 创建新的视觉点和特征
      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      // 设置点的属性
      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      // 确保法向量朝向相机
      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }
      
      pt_new->previous_normal_ = pt_new->normal_;

      // 将点插入体素地图
      insertPointIntoVoxelMap(pt_new);
      add += 1;
    }
  }

  printf("[ VIO ] Append %d new visual map points\n", add);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  // 如果没有点需要更新就直接返回
  if (total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_; // 获取当前帧的位姿
  
  // 遍历所有点
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    // 如果点已经收敛,删除非参考特征并跳过
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    // 将3D点投影到当前帧图像平面
    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;
    
    // 提取图像块
    float *patch_temp = new float[patch_size_total];
    getImagePatch(img, pc, patch_temp, 0);

    // 获取该点最后一次观测的特征
    Feature *last_feature = pt->obs_.back();
    // 如果距离上次观测超过10帧就更新(已注释)
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // 计算位姿变化
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm(); // 平移变化
    double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1)); // 旋转变化
    // 如果位姿变化超过阈值就更新
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // 计算像素距离变化
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    // 如果像素距离变化超过40就更新
    if (pixel_dist > 40) add_flag = true;

    // 限制每个3D点的观测特征数量不超过30
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
    }

    // 如果需要更新,创建新的特征观测
    if (add_flag)
    {
      update_num += 1;
      update_flag[i] = 1;
      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

// 更新参考图像块的函数,输入参数为平面地图
void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  // 如果总点数为0则直接返回
  if (total_points == 0) return;

  // 遍历所有视觉特征点
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    // 跳过未初始化法向量、已收敛、观测次数少于5次、未更新的点
    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue; 
    if (update_flag[i] == 0) continue;

    // 计算点在体素地图中的位置
    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    
    // 在平面地图中查找对应的体素
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      
      // 如果找到平面,计算点到平面的距离
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        
        // 如果点在平面范围内
        if (range_dis <= 3 * plane.radius_)
        {
          // 计算雅可比矩阵和协方差
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          // 如果点到平面距离在3倍标准差内,更新法向量
          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // 根据前一次法向量的方向确定当前法向量的方向
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();
            pt->previous_normal_ = pt->normal_;

            // 如果法向量变化很小且观测次数足够多,则认为收敛
            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
            }
          }
        }
      }
    }

    // 为每个点选择最佳的参考图像块
    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      // 计算观测方向与法向量的夹角
      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);

      // 计算图像块的均值
      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      // 计算与其他观测的NCC(归一化互相关)
      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        // 计算NCC
        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      // 计算最终得分(NCC + 夹角余弦值)
      NCC = NCC / count;
      score = NCC + cos_angle;
      ref_patch_temp->score_ = score;

      // 更新最高得分的参考图像块
      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }
  }
}

// 从参考帧投影图像块到当前帧
void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  // 如果没有特征点则直接返回
  if (total_points == 0) return;

  // 设置图像块大小和输出目录
  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  // 创建结果图像矩阵
  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);  // 普通投影结果
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1); // 考虑法向量的投影结果
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1); // 密集投影结果
  cv::Mat img_photometric_error = new_frame_->img_.clone(); // 光度误差图像

  // 获取结果图像的数据指针
  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  // 定义像素成员结构体,存储像素位置和灰度值
  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  // 遍历所有特征点
  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    // 如果特征点已初始化法向量
    if (pt->is_normal_initialized_)
    {
      // 获取参考帧特征点
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      
      // 计算特征点在当前帧和先验位姿下的投影位置
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      // 计算法向量和特征点在参考帧下的位置
      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      // 确保法向量朝向正确
      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // 获取当前帧和参考帧的图像
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      // 计算当前帧到参考帧的变换
      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      // 计算仿射变换矩阵
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // 获取最佳搜索层级
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      // 如果变形太大则跳过
      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      // 创建可视化结果
      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      // 获取当前帧的图像块
      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      // 计算光度误差
      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      // 添加可视化文字信息
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      // 在图像上绘制文字
      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);

      // 在图像上绘制矩形框和圆点
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);

      // 保存可视化结果
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      // 存储投影后的像素信息
      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      // 遍历图像块的每个像素
      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x)
        {
          // 计算像素在参考帧中的位置
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          // 计算像素投影到当前帧的位置
          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            // 存储投影后的像素信息
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      // 计算投影区域的边界
      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }

      // 计算投影区域的整数边界
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);

      // 计算仿射变换的逆矩阵
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();

      // 对投影区域内的每个像素进行处理
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          
          // 检查像素是否在图像块范围内
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            // 计算对应的参考帧像素位置和值
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }

  // 第二次遍历所有特征点,使用仿射变换
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    // 计算仿射变换矩阵
    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;

    // 对图像块中的每个像素进行投影
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }

  // 创建最终的可视化结果
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  // 合并结果图像
  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  // 转换为彩色图像
  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  
  // 计算光度误差
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  // 保存结果
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

// 预计算参考帧图像块的雅可比矩阵
void VIOManager::precomputeReferencePatches(int level)
{
  // 记录开始时间
  double t1 = omp_get_wtime();
  
  // 如果没有特征点则直接返回
  if (total_points == 0) return;

  // 定义雅可比矩阵
  MD(1, 2) Jimg;  // 图像梯度雅可比
  MD(2, 3) Jdpi;  // 投影雅可比
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;  // 位姿相关的雅可比

  // 计算总的雅可比矩阵维度
  const int H_DIM = total_points * patch_size_total;

  // 初始化雅可比矩阵
  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  // 遍历所有特征点
  for (int i = 0; i < total_points; i++)
  {
    // 计算当前金字塔层级的尺度
    const int scale = (1 << level);

    // 获取特征点和其参考帧图像
    VisualPoint *pt = visual_submap->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_;

    if (pt == nullptr) continue;

    // 计算特征点深度和在参考帧下的位置
    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotation_matrix();

    // 计算投影雅可比和反对称矩阵
    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    // 计算双线性插值的权重
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    // 遍历图像块中的每个像素
    for (int x = 0; x < patch_size; x++)
    {
      // 计算图像指针位置
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        // 计算图像梯度(x方向和y方向)
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        // 构建图像梯度雅可比
        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        // 计算旋转和平移的雅可比
        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        // 将雅可比存入总的雅可比矩阵
        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  
  // 标记参考帧图像块缓存已计算
  has_ref_patch_cache = true;
}

// 更新状态(逆解)
void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  // 如果没有特征点则直接返回
  if (total_points == 0) return;

  // 保存旧状态
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break; 
  }
}

// 更新状态(正解)
void VIOManager::updateState(cv::Mat img, int level)
{
  // 如果没有特征点则直接返回
  if (total_points == 0) return;

  // 保存旧状态
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();
    
    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);
  
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      V2D pc = cam->world2cam(pf);

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          n_meas += 1;
          
          if (exposure_estimate_en) { H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; }
          else { H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;
    
    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

// 绘制跟踪的特征点
void VIOManager::plotTrackedPoints()
{
  // 如果没有特征点则直接返回
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0) return;

  // 遍历所有特征点
  for (int i = 0; i < total_points; i++)
  {
    // 获取特征点
    VisualPoint *pt = visual_submap->voxel_points[i];
    // 计算特征点在当前帧图像中的投影位置
    V2D pc(new_frame_->w2c(pt->pos_));

    // 根据光度误差判断特征点是否为内点
    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // 如果是内点,绘制绿色圆圈
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // 绿色表示稀疏对齐跟踪成功
    }
    else
    {
      // 如果是外点,绘制蓝色圆圈
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // 蓝色表示稀疏对齐跟踪失败
    }
  }
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

// 保存数据用于Colmap
void VIOManager::dumpDataForColmap()
{
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";
  
  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(new_frame_->T_f_w_.rotation_matrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

// 处理每一帧图像的主函数
void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  // 检查并调整图像尺寸
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }

  // 保存RGB图像和拷贝图像用于可视化
  img_rgb = img.clone();
  img_cp = img.clone();

  // 如果是彩色图像则转换为灰度图
  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  // 创建新的帧对象并更新状态
  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  
  // 重置特征点网格
  resetGrid();

  double t1 = omp_get_wtime();

  // 从稀疏地图中检索特征点
  retrieveFromVisualSparseMap(img, pg, feat_map);

  double t2 = omp_get_wtime();

  // 计算雅可比矩阵并更新EKF
  computeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();

  // 生成新的地图点
  generateVisualMapPoints(img, pg);

  double t4 = omp_get_wtime();
  
  // 绘制跟踪的特征点
  plotTrackedPoints();

  // 如果需要可视化,则投影参考帧图像块到当前帧
  if (plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  // 更新地图点
  updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  // 更新参考帧图像块
  updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();
  
  // 如果需要输出到Colmap,则保存数据
  if(colmap_output_en)  dumpDataForColmap();

  // 更新帧计数和平均处理时间
  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // 打印各个阶段的处理时间统计信息
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
}