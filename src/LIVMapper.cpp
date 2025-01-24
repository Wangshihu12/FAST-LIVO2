/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),
      extR(M3D::Identity())
{
  // 初始化外参矩阵
  extrinT.assign(3, 0.0);       // IMU到LiDAR的平移向量,3个元素初始化为0
  extrinR.assign(9, 0.0);       // IMU到LiDAR的旋转矩阵,9个元素初始化为0
  cameraextrinT.assign(3, 0.0); // 相机外参平移向量
  cameraextrinR.assign(9, 0.0); // 相机外参旋转矩阵

  // 创建预处理和IMU处理对象
  p_pre.reset(new Preprocess()); // 点云预处理模块
  p_imu.reset(new ImuProcess()); // IMU数据处理模块

  // 读取参数配置
  readParameters(nh);                // 从ROS参数服务器读取各项参数
  VoxelMapConfig voxel_config;       // 体素地图配置
  loadVoxelConfig(nh, voxel_config); // 加载体素地图参数

  // 初始化点云数据结构
  visual_sub_map.reset(new PointCloudXYZI());   // 视觉子地图点云
  feats_undistort.reset(new PointCloudXYZI());  // 去畸变后的特征点云
  feats_down_body.reset(new PointCloudXYZI());  // 降采样后机体系下的点云
  feats_down_world.reset(new PointCloudXYZI()); // 降采样后世界系下的点云
  pcl_w_wait_pub.reset(new PointCloudXYZI());   // 等待发布的世界系点云
  pcl_wait_pub.reset(new PointCloudXYZI());     // 等待发布的点云
  pcl_wait_save.reset(new PointCloudXYZRGB());  // 等待保存的RGB点云

  // 创建地图管理器和VIO管理器
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map)); // 体素地图管理器
  vio_manager.reset(new VIOManager());                                  // 视觉惯性里程计管理器

  // 其他初始化
  root_dir = ROOT_DIR;    // 设置根目录
  initializeFiles();      // 初始化文件
  initializeComponents(); // 初始化组件

  // 设置路径信息
  path.header.stamp = ros::Time::now(); // 设置时间戳
  path.header.frame_id = "camera_init"; // 设置坐标系
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

// 初始化各个组件
void LIVMapper::initializeComponents()
{
  // 设置点云降采样滤波器的叶子大小
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

  // 设置IMU到LiDAR的外参矩阵
  extT << VEC_FROM_ARRAY(extrinT); // 平移向量
  extR << MAT_FROM_ARRAY(extrinR); // 旋转矩阵

  // 设置体素地图管理器的外参
  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  // 从ROS参数服务器加载相机模型参数
  if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam))
    throw std::runtime_error("Camera model not correctly specified.");

  // 配置VIO管理器的参数
  vio_manager->grid_size = grid_size;                                   // 图像网格大小
  vio_manager->patch_size = patch_size;                                 // 特征块大小
  vio_manager->outlier_threshold = outlier_threshold;                   // 外点阈值
  vio_manager->setImuToLidarExtrinsic(extT, extR);                      // 设置IMU到LiDAR外参
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT); // 设置LiDAR到相机外参
  vio_manager->state = &_state;                                         // 设置状态指针
  vio_manager->state_propagat = &state_propagat;                        // 设置状态传播指针
  vio_manager->max_iterations = max_iterations;                         // 最大迭代次数
  vio_manager->img_point_cov = IMG_POINT_COV;                           // 图像点协方差
  vio_manager->normal_en = normal_en;                                   // 是否启用法向量
  vio_manager->inverse_composition_en = inverse_composition_en;         // 是否启用逆向组合
  vio_manager->raycast_en = raycast_en;                                 // 是否启用光线投射
  vio_manager->grid_n_width = grid_n_width;                             // 网格宽度
  vio_manager->grid_n_height = grid_n_height;                           // 网格高度
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;               // 图像金字塔层数
  vio_manager->exposure_estimate_en = exposure_estimate_en;             // 是否估计曝光时间
  vio_manager->colmap_output_en = colmap_output_en;                     // 是否输出COLMAP格式
  vio_manager->initializeVIO();                                         // 初始化VIO

  // 配置IMU处理器参数
  p_imu->set_extrinsic(extT, extR);                         // 设置外参
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov)); // 陀螺仪协方差
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov)); // 加速度计协方差
  p_imu->set_inv_expo_cov(inv_expo_cov);                    // 曝光时间协方差
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));     // 陀螺仪偏置协方差
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));     // 加速度计偏置协方差
  p_imu->set_imu_init_frame_num(imu_int_frame);             // IMU初始化帧数

  // 根据配置禁用相应功能
  if (!imu_en)
    p_imu->disable_imu(); // 禁用IMU
  if (!gravity_est_en)
    p_imu->disable_gravity_est(); // 禁用重力估计
  if (!ba_bg_est_en)
    p_imu->disable_bias_est(); // 禁用偏置估计
  if (!exposure_estimate_en)
    p_imu->disable_exposure_est(); // 禁用曝光估计

  // 设置SLAM模式:LIVO(视觉+惯性+激光)、ONLY_LIO(仅惯性+激光)或ONLY_LO(仅激光)
  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO
                                                    : ONLY_LO;
}

// 此函数用于初始化和打开程序所需的各种输出文件

void LIVMapper::initializeFiles()
{
  // 如果同时启用了点云保存和COLMAP输出功能
  if (pcd_save_en && colmap_output_en)
  {
    // 构建colmap输出脚本的路径
    const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";

    // 构建chmod命令,给脚本添加可执行权限
    std::string chmodCommand = "chmod +x " + folderPath;

    // 执行chmod命令
    int chmodRet = system(chmodCommand.c_str());
    if (chmodRet != 0)
    {
      std::cerr << "Failed to set execute permissions for the script." << std::endl;
      return;
    }

    // 执行colmap输出脚本
    int executionRet = system(folderPath.c_str());
    if (executionRet != 0)
    {
      std::cerr << "Failed to execute the script." << std::endl;
      return;
    }
  }

  // 如果启用了COLMAP输出,打开3D点输出文件
  if (colmap_output_en)
    fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);

  // 如果设置了点云保存间隔,打开点云位姿输出文件
  if (pcd_save_interval > 0)
    fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);

  // 打开调试用的矩阵输出文件
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it)
{
  sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this) : nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  sub_img = nh.subscribe(img_topic, 200000, &LIVMapper::img_cbk, this);

  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImage = it.advertise("/rgb_img", 1);
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
}

// 处理第一帧激光雷达数据的函数
void LIVMapper::handleFirstFrame()
{
  // 如果还没有处理过第一帧
  if (!is_first_frame)
  {
    // 记录第一帧激光雷达的时间戳
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    // 将时间戳传给IMU处理模块(仅用于IMU数据记录)
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    // 标记已处理第一帧
    is_first_frame = true;
    // 打印提示信息
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

// 重力对齐函数
void LIVMapper::gravityAlignment()
{
  // 只有当IMU初始化完成且重力对齐未完成时才执行
  if (!p_imu->imu_need_init && !gravity_align_finished)
  {
    std::cout << "Gravity Alignment Starts" << std::endl;

    // 定义世界坐标系的重力方向(向下为负)和IMU测量的重力方向
    V3D ez(0, 0, -1), gz(_state.gravity);

    // 计算从IMU坐标系到世界坐标系的旋转四元数
    // FromTwoVectors计算将gz旋转到ez的最小旋转
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);

    // 将四元数转换为旋转矩阵
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    // 将状态量从IMU坐标系转换到世界坐标系
    _state.pos_end = G_R_I0 * _state.pos_end; // 位置
    _state.rot_end = G_R_I0 * _state.rot_end; // 旋转
    _state.vel_end = G_R_I0 * _state.vel_end; // 速度
    _state.gravity = G_R_I0 * _state.gravity; // 重力

    // 标记重力对齐完成
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu()
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en)
    gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

// 状态估计和建图函数
void LIVMapper::stateEstimationAndMapping()
{
  // 根据LidarMeasures中的标志位判断使用VIO还是LIO
  switch (LidarMeasures.lio_vio_flg)
  {
  case VIO:      // 如果是视觉惯性里程计(VIO)模式
    handleVIO(); // 调用VIO处理函数
    break;
  case LIO:      // 如果是激光惯性里程计(LIO)模式
    handleLIO(); // 调用LIO处理函数
    break;
  }
}

// VIO处理函数
void LIVMapper::handleVIO()
{
  // 计算当前姿态的欧拉角
  euler_cur = RotMtoEuler(_state.rot_end);

  // 将当前状态写入预测文件
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;

  // 检查点云数据是否为空
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr))
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }

  // 输出原始特征点数量
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  // 根据时间判断是否需要绘图
  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1))
  {
    vio_manager->plot_flag = true;
  }
  else
  {
    vio_manager->plot_flag = false;
  }

  // 处理当前帧图像
  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  // IMU预测使能时,更新EKF状态
  if (imu_prop_enable)
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // 发布世界坐标系下的点云和RGB图像
  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  // 将当前状态写入输出文件
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

// handleLIO() 函数实现了LIO(激光惯性里程计)的主要处理流程
void LIVMapper::handleLIO()
{
  // 计算当前姿态的欧拉角
  euler_cur = RotMtoEuler(_state.rot_end);

  // 将当前状态写入预测文件,包括时间、欧拉角、位置、速度、陀螺仪偏置、加速度计偏置等
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;

  // 检查点云数据是否为空
  if (feats_undistort->empty() || (feats_undistort == nullptr))
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  // 对原始点云进行降采样
  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);

  double t_down = omp_get_wtime();

  // 更新降采样后的点云信息
  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  // 将点云从机体坐标系转换到世界坐标系
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;

  // 首次构建体素地图
  if (!lidar_map_inited)
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  // 进行状态估计
  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  // IMU预测使能时,更新EKF状态
  if (imu_prop_enable)
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // 如果需要输出位姿信息
  if (pose_output_en)
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    // 打开或追加写入位姿文件
    if (!pos_opend)
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open())
        ROS_ERROR("open fail\n");
    }
    else
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open())
        ROS_ERROR("open fail\n");
    }
    // 将位姿信息写入文件
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }

  // 发布里程计信息
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  // 更新点云在世界坐标系下的位置和协方差
  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++)
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  // 更新体素地图
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;

  double t4 = omp_get_wtime();

  // 如果启用了地图滑动窗口
  if (voxelmap_manager->config_setting_.map_sliding_en)
  {
    if (voxelmap_manager->mapSliding())
    {
      // update_local_voxel_map();
    }
    // publish_local_voxelmap(local_voxel_clouds_publisher);
  }

  // 处理点云数据并发布
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  // 发布各种消息
  if (!img_en)
    publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en)
    publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  // 计算并输出各阶段耗时统计
  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // 打印漂亮的时间统计表格
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  // 将最终状态写入输出文件
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD()
{
  if (pcd_save_en && pcl_wait_save->points.size() > 0 && pcd_save_interval < 0)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(pcl_wait_save);
    voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
    voxel_filter.filter(*downsampled_cloud);

    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";

    pcl::PCDWriter pcd_writer;

    // Save the raw point cloud data
    pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save);
    std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir
              << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;

    // Save the downsampled point cloud data
    pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud);
    std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir
              << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

    if (colmap_output_en)
    {
      fout_points << "# 3D point list with one line of data per point\n";
      fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
      for (size_t i = 0; i < downsampled_cloud->size(); ++i)
      {
        const auto &point = downsampled_cloud->points[i];
        fout_points << i << " "
                    << std::fixed << std::setprecision(6)
                    << point.x << " " << point.y << " " << point.z << " "
                    << static_cast<int>(point.r) << " "
                    << static_cast<int>(point.g) << " "
                    << static_cast<int>(point.b) << " "
                    << 0 << std::endl;
      }
    }
  }
}

void LIVMapper::run()
{
  ros::Rate rate(5000);
  while (ros::ok())
  {
    ros::spinOnce();
    if (!sync_packages(LidarMeasures))
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }
  savePCD();
}

// 这个函数实现了IMU状态的单步传播
// 输入参数:
// - imu_prop_state: IMU状态量,包含位置、速度、姿态、偏置等
// - dt: 时间间隔
// - acc_avr: 加速度计测量值
// - angvel_avr: 陀螺仪测量值
void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  // 对加速度计测量值进行归一化处理,并减去偏置
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  // 陀螺仪测量值减去偏置
  angvel_avr -= imu_prop_state.bias_g;

  // 计算角速度的指数映射矩阵
  M3D Exp_f = Exp(angvel_avr, dt);
  /* 更新IMU姿态 */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* 计算IMU在世界坐标系下的加速度 */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* 更新IMU位置,使用匀加速运动公式 */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* 更新IMU速度 */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

// IMU状态传播回调函数,用于实时更新IMU状态
void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  // 如果IMU需要初始化、没有新的IMU数据或EKF未完成一次迭代,则直接返回
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once)
  {
    return;
  }

  // 加锁保护IMU缓冲区
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致

  // 如果IMU传播使能且IMU缓冲区非空
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;

    // 如果状态更新标志为真
    if (state_update_flg)
    {
      // 使用最新的EKF状态初始化IMU传播状态
      imu_propagate = latest_ekf_state;

      // 丢弃所有过时的IMU数据包
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }

      last_t_from_lidar_end_time = 0;

      // 对IMU缓冲区中的每个IMU数据进行状态传播
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;

        // 获取IMU加速度和角速度测量值
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);

        // 进行一次IMU状态传播
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      // 使用最新的IMU数据进行状态传播
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    // 发布IMU传播后的里程计信息
    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);

    // 设置里程计消息
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();

    // 发布里程计消息
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T>
void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T>
Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en)
    return;
  mtx_buffer.lock();
  // cout<<"got feature"<<endl;
  if (msg->header.stamp.toSec() < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(msg->header.stamp.toSec());
  last_timestamp_lidar = msg->header.stamp.toSec();

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en)
    return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en)
    return;

  if (last_timestamp_lidar < 0.0)
    return;
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en)
    timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  {

    ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init)
    {
      prop_imu_buffer.push_back(*msg);
    }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

// static int i = 0;
void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en)
    return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  // if (hilti_en)
  // {
  //   i++;
  //   if (i % 4 != 0) return;
  // }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001)
    return;
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0)
    return;

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (lid_raw_data_buffer.empty() && lidar_en)
    return false;
  if (img_buffer.empty() && img_en)
    return false;
  if (imu_buffer.empty() && imu_en)
    return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0)
      meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1)
        return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                 // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true; // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time)
        break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // 获取图像捕获时间 = 图像时间戳 + 曝光时间
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      
      // 如果是第一次LIO更新,使用第一帧激光雷达时间作为上次更新时间
      if (meas.last_lio_update_time < 0.0)
        meas.last_lio_update_time = lid_header_time_buffer.front();

      // 获取最新的激光雷达帧结束时间和IMU时间
      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = imu_buffer.back()->header.stamp.toSec();

      // 如果图像时间早于上次LIO更新时间,丢弃该帧图像
      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      // 如果图像时间晚于最新的激光雷达或IMU数据,返回false等待新数据
      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        return false;
      }

      struct MeasureGroup m;

      // 清空并填充IMU数据
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (imu_buffer.front()->header.stamp.toSec() > m.lio_time)
          break;

        // 只保留上次LIO更新之后的IMU数据
        if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time)
          m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      // 交换当前点云和下一帧点云的数据
      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      // 预分配点云内存
      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);

      // 处理激光雷达数据
      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time)
          break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        // 根据点的时间戳将点分配到当前帧或下一帧
        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            // 早于图像时间的点放入当前帧
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            // 晚于图像时间的点放入下一帧
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      // 保存测量数据并设置标志为LIO模式
      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      return true;
    }

    case LIO:
    {
      // 计算图像捕获时间 = 图像时间戳 + 曝光时间初值
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      
      // 设置测量标志为VIO模式
      meas.lio_vio_flg = VIO;
      
      // 清空测量数据
      meas.measures.clear();
      
      // 获取IMU数据的时间戳
      double imu_time = imu_buffer.front()->header.stamp.toSec();

      // 创建测量组结构体
      struct MeasureGroup m;
      m.vio_time = img_capture_time;  // 设置VIO时间为图像捕获时间
      m.lio_time = meas.last_lio_update_time;  // 设置LIO时间为上次LIO更新时间
      m.img = img_buffer.front();  // 获取图像数据
      
      // 加锁以保护共享数据
      mtx_buffer.lock();
      
      // 注释掉的代码块用于同步IMU数据
      // 当IMU缓冲区非空且IMU时间早于图像捕获时间时:
      // - 获取IMU时间戳
      // - 如果IMU时间晚于图像捕获时间则跳出
      // - 将IMU数据加入测量组
      // - 从IMU缓冲区移除已处理的数据
      // - 打印IMU时间信息
      
      // 从缓冲区移除已处理的图像数据
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      
      // 解锁
      mtx_buffer.unlock();
      
      // 通知所有等待的线程
      sig_buffer.notify_all();
      
      // 将测量组加入测量数据
      meas.measures.push_back(m);
      
      // VIO更新后需要刷新激光雷达帧结束时间,因此设置lidar_pushed为false
      lidar_pushed = false;
      
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty())
    return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en)
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if (pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w));
        if (pf[2] < 0)
          continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255;
          // else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255;
          // else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255;
          // else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points)
            laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
    else
    {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en)
  {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else
  {
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg);
  }
  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en)
  {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;

    *pcl_wait_save += *laserCloudWorldRGB;
    scan_wait_num++;

    if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
        PointCloudXYZRGB().swap(*pcl_wait_save);
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                     << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if (laserCloudWorldRGB->size() > 0)
  {
    PointCloudXYZI().swap(*pcl_wait_pub);
    PointCloudXYZRGB().swap(*laserCloudWorldRGB);
  }
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size();
  if (size == 0)
    return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T>
void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped"));
  pubOdomAftMapped.publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}