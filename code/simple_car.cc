// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/simple_car/simple_car.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc {

std::string SimpleCar::XmlPath() const {
  return GetModelPath("simple_car/task.xml");
}

std::string SimpleCar::Name() const { return "SimpleCar"; }


// ------- Residuals for simple_car task ------
//     Position: Car should reach goal position (x, y)
//     Control:  Controls should be small
// ------------------------------------------
void SimpleCar::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                     double* residual) const {
  // ---------- Position (x, y) ----------
  // Goal position from mocap body
  residual[0] = data->qpos[0] - data->mocap_pos[0];  // x position
  residual[1] = data->qpos[1] - data->mocap_pos[1];  // y position

  // ---------- Control ----------
  residual[2] = data->ctrl[0];  // forward control
  residual[3] = data->ctrl[1];  // turn control
}

// -------- Transition for simple_car task --------
//   If car is within tolerance of goal ->
//   move goal randomly.
// ------------------------------------------------
void SimpleCar::TransitionLocked(mjModel* model, mjData* data) {
  // Car position (x, y)
  double car_pos[2] = {data->qpos[0], data->qpos[1]};
  
  // Goal position from mocap
  double goal_pos[2] = {data->mocap_pos[0], data->mocap_pos[1]};
  
  // Distance to goal
  double car_to_goal[2];
  mju_sub(car_to_goal, goal_pos, car_pos, 2);
  
  // If within tolerance, move goal to random position
  if (mju_norm(car_to_goal, 2) < 0.2) {
    absl::BitGen gen_;
    data->mocap_pos[0] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[1] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[2] = 0.01;  // keep z at ground level
  }
}

// draw task-related geometry in the scene
// 改进后的立式仪表盘，放置在汽车正上方

void SimpleCar::ModifyScene(const mjModel* model, const mjData* data,
                            mjvScene* scene) const {
  static int filter_initialized = 0;
  static double speed_kmh_filtered = 0.0;
  static double wheel_rpm_filtered = 0.0;
  static double throttle_filtered = 0.0;
  static double steer_filtered = 0.0;

  auto clamp = [](double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
  };

  double dt = model->opt.timestep;
  if (dt <= 0.0) dt = 0.001;

  double throttle_raw = (model->nu > 0) ? data->ctrl[0] : 0.0;
  double steer_raw = (model->nu > 1) ? data->ctrl[1] : 0.0;

  double throttle_min = -1.0, throttle_max = 1.0;
  double steer_min = -1.0, steer_max = 1.0;
  if (model->actuator_ctrlrange && model->nu >= 2) {
    throttle_min = model->actuator_ctrlrange[0 * 2 + 0];
    throttle_max = model->actuator_ctrlrange[0 * 2 + 1];
    steer_min = model->actuator_ctrlrange[1 * 2 + 0];
    steer_max = model->actuator_ctrlrange[1 * 2 + 1];
  }

  double throttle_centered = throttle_raw;
  if (throttle_max > throttle_min) {
    double mid = 0.5 * (throttle_min + throttle_max);
    double half = 0.5 * (throttle_max - throttle_min);
    throttle_centered = (half > 1e-9) ? (throttle_raw - mid) / half : 0.0;
  }
  throttle_centered = clamp(throttle_centered, -1.0, 1.0);

  double steer_centered = steer_raw;
  if (steer_max > steer_min) {
    double mid = 0.5 * (steer_min + steer_max);
    double half = 0.5 * (steer_max - steer_min);
    steer_centered = (half > 1e-9) ? (steer_raw - mid) / half : 0.0;
  }
  steer_centered = clamp(steer_centered, -1.0, 1.0);

  // 5分钟从100%降到0%的油量消耗逻辑
  const double total_time_minutes = 5.0;
  const double total_time_seconds = total_time_minutes * 60.0;
  
  // 根据游戏运行时间计算当前油量百分比
  double fuel_percent = 100.0 - (data->time / total_time_seconds) * 100.0;
  fuel_percent = clamp(fuel_percent, 0.0, 100.0);
  

  int car_body_id = mj_name2id(model, mjOBJ_BODY, "car");
  if (car_body_id < 0) return;

  auto SensorPtr = [&](const char* name) -> const double* {
    int id = mj_name2id(model, mjOBJ_SENSOR, name);
    if (id < 0) return nullptr;
    int adr = model->sensor_adr[id];
    if (adr < 0) return nullptr;
    return data->sensordata + adr;
  };

  double speed_ms = 0.0;
  if (const double* car_velocity = SensorPtr("car_velocity")) {
    speed_ms = std::sqrt(car_velocity[0] * car_velocity[0] +
                         car_velocity[1] * car_velocity[1] +
                         car_velocity[2] * car_velocity[2]);
  } else {
    const mjtNum* cvel = data->cvel + 6 * car_body_id;
    speed_ms = std::sqrt(static_cast<double>(cvel[0] * cvel[0] +
                                             cvel[1] * cvel[1] +
                                             cvel[2] * cvel[2]));
  }

  double speed_kmh = speed_ms * 3.6;

  double wheel_rpm = 0.0;
  {
    int left_joint_id = mj_name2id(model, mjOBJ_JOINT, "left");
    int right_joint_id = mj_name2id(model, mjOBJ_JOINT, "right");
    double omega = 0.0;
    int count = 0;
    if (left_joint_id >= 0) {
      int dof = model->jnt_dofadr[left_joint_id];
      if (dof >= 0 && dof < model->nv) {
        omega += std::abs(static_cast<double>(data->qvel[dof]));
        count++;
      }
    }
    if (right_joint_id >= 0) {
      int dof = model->jnt_dofadr[right_joint_id];
      if (dof >= 0 && dof < model->nv) {
        omega += std::abs(static_cast<double>(data->qvel[dof]));
        count++;
      }
    }
    if (count > 0) {
      omega /= count;
    } else if (const double* car_angular_velocity =
                   SensorPtr("car_angular_velocity")) {
      omega = std::sqrt(car_angular_velocity[0] * car_angular_velocity[0] +
                        car_angular_velocity[1] * car_angular_velocity[1] +
                        car_angular_velocity[2] * car_angular_velocity[2]);
    }
    wheel_rpm = omega * 60.0 / (2.0 * 3.141592653589793);
  }
  
  // 获取汽车位置
  double* car_pos = data->xpos + 3 * car_body_id;
  
  // 实时打印小车位置、速度等数据（在同一行更新）
  printf("\rCar Position: (%.2f, %.2f, %.2f) | Speed: %.2f km/h | Fuel: %.1f%% ", 
         car_pos[0], car_pos[1], car_pos[2], speed_kmh_filtered, fuel_percent);
  fflush(stdout); // 确保立即输出

  const double* car_xmat = data->xmat + 9 * car_body_id;
  mjtNum car_forward_mjt[3] = {static_cast<mjtNum>(car_xmat[0]),
                               static_cast<mjtNum>(car_xmat[3]),
                               static_cast<mjtNum>(car_xmat[6])};
  if (mju_norm3(car_forward_mjt) > static_cast<mjtNum>(1e-9)) {
    mju_normalize3(car_forward_mjt);
  } else {
    car_forward_mjt[0] = static_cast<mjtNum>(1.0);
    car_forward_mjt[1] = static_cast<mjtNum>(0.0);
    car_forward_mjt[2] = static_cast<mjtNum>(0.0);
  }
  float car_forward_world[3] = {static_cast<float>(car_forward_mjt[0]),
                                static_cast<float>(car_forward_mjt[1]),
                                static_cast<float>(car_forward_mjt[2])};
  
  // 仪表盘位置（汽车正前方，立起来）
  float dashboard_pos[3] = {
    static_cast<float>(car_pos[0]) + 0.25f * car_forward_world[0],
    static_cast<float>(car_pos[1]) + 0.25f * car_forward_world[1],
    static_cast<float>(car_pos[2] + 0.30f) + 0.25f * car_forward_world[2]
  };
    const float gauge_scale = 0.7f;  // 仪表盘整体缩小至0.7倍


  // 最大速度参考值（km/h），根据要求是0-10
  const float max_speed_kmh = 10.0f;
  
  const double tau = 0.18;
  const double alpha = 1.0 - std::exp(-dt / tau);
  if (!filter_initialized) {
    speed_kmh_filtered = 0.0;  // 初始速度设为0，确保指针指向0
    wheel_rpm_filtered = 0.0;
    throttle_filtered = throttle_centered;
    steer_filtered = steer_centered;
    filter_initialized = 1;
  } else {
    speed_kmh_filtered += alpha * (speed_kmh - speed_kmh_filtered);
    wheel_rpm_filtered += alpha * (wheel_rpm - wheel_rpm_filtered);
    throttle_filtered += alpha * (throttle_centered - throttle_filtered);
    steer_filtered += alpha * (steer_centered - steer_filtered);
  }

  // 速度百分比（0-1）
  float speed_ratio = static_cast<float>(speed_kmh_filtered) / max_speed_kmh;
  if (speed_ratio > 1.0f) speed_ratio = 1.0f;
  if (speed_ratio < 0.0f) speed_ratio = 0.0f;
  
  double dashboard_rot_mat[9];
  bool use_billboard = false;
  if (scene && scene->camera) {
    mjtNum cam_forward[3] = {static_cast<mjtNum>(scene->camera[0].forward[0]),
                             static_cast<mjtNum>(scene->camera[0].forward[1]),
                             static_cast<mjtNum>(scene->camera[0].forward[2])};
    mjtNum cam_up[3] = {static_cast<mjtNum>(scene->camera[0].up[0]),
                        static_cast<mjtNum>(scene->camera[0].up[1]),
                        static_cast<mjtNum>(scene->camera[0].up[2])};

    mjtNum right[3];
    mju_cross(right, cam_forward, cam_up);

    mjtNum right_norm = mju_norm3(right);
    mjtNum up_norm = mju_norm3(cam_up);
    mjtNum forward_norm = mju_norm3(cam_forward);

    if (right_norm > 1e-9 && up_norm > 1e-9 && forward_norm > 1e-9) {
      mju_scl3(right, right, 1.0 / right_norm);
      mju_scl3(cam_up, cam_up, 1.0 / up_norm);
      mju_scl3(cam_forward, cam_forward, 1.0 / forward_norm);

      mjtNum normal[3] = {-cam_forward[0], -cam_forward[1], -cam_forward[2]};

      dashboard_rot_mat[0] = right[0];
      dashboard_rot_mat[1] = cam_up[0];
      dashboard_rot_mat[2] = normal[0];
      dashboard_rot_mat[3] = right[1];
      dashboard_rot_mat[4] = cam_up[1];
      dashboard_rot_mat[5] = normal[1];
      dashboard_rot_mat[6] = right[2];
      dashboard_rot_mat[7] = cam_up[2];
      dashboard_rot_mat[8] = normal[2];

      use_billboard = true;
    }
  }

  if (!use_billboard) {
    double angle_x = 90.0 * 3.141592653589793 / 180.0;
    double cos_x = cos(angle_x);
    double sin_x = sin(angle_x);
    double mat_x[9] = {1, 0, 0, 0, cos_x, -sin_x, 0, sin_x, cos_x};

    double angle_z = -90.0 * 3.141592653589793 / 180.0;
    double cos_z = cos(angle_z);
    double sin_z = sin(angle_z);
    double mat_z[9] = {cos_z, -sin_z, 0, sin_z, cos_z, 0, 0, 0, 1};

    mju_mulMatMat(dashboard_rot_mat, mat_z, mat_x, 3, 3, 3);
  }
  
  const float x_axis_world[3] = {static_cast<float>(dashboard_rot_mat[0]),
                                 static_cast<float>(dashboard_rot_mat[3]),
                                 static_cast<float>(dashboard_rot_mat[6])};
  const float y_axis_world[3] = {static_cast<float>(dashboard_rot_mat[1]),
                                 static_cast<float>(dashboard_rot_mat[4]),
                                 static_cast<float>(dashboard_rot_mat[7])};
  const float z_axis_world[3] = {static_cast<float>(dashboard_rot_mat[2]),
                                 static_cast<float>(dashboard_rot_mat[5]),
                                 static_cast<float>(dashboard_rot_mat[8])};

  auto Place = [&](const float origin[3], float x, float y, float z,
                   float out[3]) {
    out[0] = origin[0] + x * x_axis_world[0] + y * y_axis_world[0] +
             z * z_axis_world[0];
    out[1] = origin[1] + x * x_axis_world[1] + y * y_axis_world[1] +
             z * z_axis_world[1];
    out[2] = origin[2] + x * x_axis_world[2] + y * y_axis_world[2] +
             z * z_axis_world[2];
  };

  auto PushGeom = [&](mjtGeom type, const float size_f[3], const float pos_f[3],
                      const double mat_d[9],
                      const float rgba[4]) -> mjvGeom* {
    if (scene->ngeom >= scene->maxgeom) return nullptr;
    mjtNum size[3] = {static_cast<mjtNum>(size_f[0]),
                      static_cast<mjtNum>(size_f[1]),
                      static_cast<mjtNum>(size_f[2])};
    mjtNum pos[3] = {static_cast<mjtNum>(pos_f[0]), static_cast<mjtNum>(pos_f[1]),
                     static_cast<mjtNum>(pos_f[2])};
    mjtNum mat[9];
    for (int i = 0; i < 9; i++) mat[i] = static_cast<mjtNum>(mat_d[i]);
    mjvGeom* geom = scene->geoms + scene->ngeom;
    mjv_initGeom(geom, type, size, pos, mat, rgba);
    geom->category = mjCAT_DECOR;
    scene->ngeom++;
    return geom;
  };

  auto PushLabel = [&](const float pos_f[3], float size_scalar,
                       const char* text, const float rgba[4]) {
    float size_f[3] = {size_scalar, size_scalar, size_scalar};
    mjvGeom* geom = PushGeom(mjGEOM_LABEL, size_f, pos_f, dashboard_rot_mat, rgba);
    if (!geom) return;
    std::strncpy(geom->label, text, sizeof(geom->label) - 1);
    geom->label[sizeof(geom->label) - 1] = '\0';
  };

  const float bezel_radius = 0.170f * gauge_scale;
  const float face_radius = 0.155f * gauge_scale;
  const float tick_outer_radius = 0.145f * gauge_scale;
  const float overlay_z = 0.026f * gauge_scale;

  {
    float size[3] = {bezel_radius * 1.03f, bezel_radius * 1.03f,
                     0.014f * gauge_scale};
    float pos[3];
    Place(dashboard_pos, 0.0f, 0.0f, -0.0018f * gauge_scale, pos);
    const float rgba[4] = {0.97f, 0.97f, 0.97f, 1.0f};
    PushGeom(mjGEOM_ELLIPSOID, size, pos, dashboard_rot_mat, rgba);
  }

  {
    float size[3] = {bezel_radius, bezel_radius, 0.022f * gauge_scale};
    const float rgba[4] = {0.18f, 0.18f, 0.19f, 0.95f};
    PushGeom(mjGEOM_ELLIPSOID, size, dashboard_pos, dashboard_rot_mat, rgba);
  }

  {
    float size[3] = {face_radius, face_radius, 0.018f * gauge_scale};
    const float rgba[4] = {0.03f, 0.03f, 0.035f, 0.96f};
    PushGeom(mjGEOM_ELLIPSOID, size, dashboard_pos, dashboard_rot_mat, rgba);
  }

  {
    float size[3] = {bezel_radius, bezel_radius, 0.024f * gauge_scale};
    float pos[3];
    Place(dashboard_pos, 0.0f, 0.0f, 0.0022f * gauge_scale, pos);
    const float rgba[4] = {0.60f, 0.75f, 0.95f, 0.06f};
    PushGeom(mjGEOM_ELLIPSOID, size, pos, dashboard_rot_mat, rgba);
  }

  const int kArcSegments = 26;
  for (int i = 0; i < kArcSegments; i++) {
    if (scene->ngeom >= scene->maxgeom) break;

    const float seg_ratio =
        (kArcSegments <= 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(kArcSegments - 1);
    float angle_deg = 180.0f * seg_ratio;
    float rad_angle = angle_deg * 3.14159f / 180.0f;

    float r = 0.18f, g = 0.85f, b = 0.30f;
    if (seg_ratio >= 0.85f) {
      r = 0.95f; g = 0.15f; b = 0.20f;
    } else if (seg_ratio >= 0.65f) {
      r = 0.98f; g = 0.78f; b = 0.10f;
    }

    float size[3] = {0.0045f * gauge_scale, 0.0085f * gauge_scale,
                     0.0030f * gauge_scale};
    float pos[3];
    Place(dashboard_pos, tick_outer_radius * cos(rad_angle),
          tick_outer_radius * sin(rad_angle), overlay_z, pos);

    double rot_angle = static_cast<double>(angle_deg) + 90.0;
    double rad_rot = rot_angle * 3.14159 / 180.0;
    double cos_t = cos(rad_rot);
    double sin_t = sin(rad_rot);
    double seg_rot_mat[9] = {cos_t, -sin_t, 0, sin_t, cos_t, 0, 0, 0, 1};

    double seg_mat[9];
    mju_mulMatMat(seg_mat, dashboard_rot_mat, seg_rot_mat, 3, 3, 3);
    const float rgba[4] = {r, g, b, 0.85f};
    PushGeom(mjGEOM_BOX, size, pos, seg_mat, rgba);
  }

    // 2. 添加刻度线（0~10 全刻度）
    const int kMaxTick = 10;
    const int kTickCount = kMaxTick + 1;

    for (int i = 0; i < kTickCount; i++) {
        if (scene->ngeom >= scene->maxgeom) break;

        int tick_value = i;
        const bool is_major = (tick_value % 5 == 0);

        // 角度：0 在右(225°)，10 在左(-45°)
        float tick_angle_deg = 225.0f - 270.0f * (static_cast<float>(tick_value) / kMaxTick);
        float rad_tick_angle = tick_angle_deg * 3.14159f / 180.0f;

        float full_len = (is_major ? 0.040f : 0.024f) * gauge_scale;
        float half_len = full_len * 0.5f;

        float tick_radius_outer = 0.138f * gauge_scale;
        float tick_radius_center = tick_radius_outer - half_len;

        float size[3] = {(is_major ? 0.0034f : 0.0026f) * gauge_scale, half_len,
                         0.0030f * gauge_scale};
        float pos[3];
        Place(dashboard_pos, tick_radius_center * cos(rad_tick_angle),
              tick_radius_center * sin(rad_tick_angle), overlay_z, pos);

        // ---- 刻度指向圆心 ----
        double tick_rot_angle = tick_angle_deg + 90.0;
        double rad_tick_rot = tick_rot_angle * 3.14159 / 180.0;
        double cos_t = cos(rad_tick_rot);
        double sin_t = sin(rad_tick_rot);

        double tick_rot_mat[9] = {
                cos_t, -sin_t, 0,
                sin_t,  cos_t, 0,
                0,      0,     1
        };

        double tick_mat[9];
        mju_mulMatMat(tick_mat, dashboard_rot_mat, tick_rot_mat, 3, 3, 3);

        float rgba[4];
        if (is_major) {
          rgba[0] = 0.95f;
          rgba[1] = 0.95f;
          rgba[2] = 0.95f;
          rgba[3] = 1.0f;
        } else {
          rgba[0] = 0.70f;
          rgba[1] = 0.70f;
          rgba[2] = 0.70f;
          rgba[3] = 0.95f;
        }
        PushGeom(mjGEOM_BOX, size, pos, tick_mat, rgba);

        // ---- 数字标签 ----
        if (scene->ngeom >= scene->maxgeom) break;
        float label_pos[3];
        float label_radius = 0.192f * gauge_scale;
        Place(dashboard_pos, label_radius * cos(rad_tick_angle),
              label_radius * sin(rad_tick_angle), overlay_z, label_pos);
        char label_text[16];
        std::snprintf(label_text, sizeof(label_text), "%d", tick_value);
        const float rgba_label[4] = {0.92f, 0.92f, 0.92f, 1.0f};
        PushLabel(label_pos, 0.047f * gauge_scale, label_text, rgba_label);
    }

    // 3. 速度指针（美化后的设计）
  if (scene->ngeom < scene->maxgeom) {
    // 计算指针角度：与刻度角度范围一致，0在右(225°)，10在左(-45°)
    float angle = 225.0f - 270.0f * speed_ratio;  // 225度到-45度范围
    // 确保初始角度正确对应刻度0
    if (speed_kmh_filtered < 0.1) {
      angle = 225.0f;
    }
    
    const float pointer_total_len = 0.120f * gauge_scale;  // 总长度
    float pos[3];
    // 将指针的旋转中心放在仪表盘中心
    Place(dashboard_pos, 0.0f, 0.0f, overlay_z + 0.0040f * gauge_scale, pos);
    
    // 指针旋转：需要绕仪表盘法线旋转，然后再应用仪表盘的旋转
    // 首先，绕Z轴旋转到指针角度（相对于仪表盘）
    double pointer_angle = angle + 90.0;  // 需要调整90度，因为指针元素是矩形，类似刻度线
    double rad_pointer_angle = pointer_angle * 3.14159 / 180.0;
    double cos_p = cos(rad_pointer_angle);
    double sin_p = sin(rad_pointer_angle);
    double pointer_rot_mat[9] = {
      cos_p, -sin_p, 0,
      sin_p,  cos_p, 0,
      0,      0,     1
    };
    
    // 组合旋转：先绕Z轴旋转到指针角度，再应用仪表盘旋转
    double temp_mat[9];
    mju_mulMatMat(temp_mat, dashboard_rot_mat, pointer_rot_mat, 3, 3, 3);

    // 指针底座（增强立体感）
    if (scene->ngeom < scene->maxgeom) {
      float size[3] = {0.0075f * gauge_scale, 0.0075f * gauge_scale,
                       0.0055f * gauge_scale};
      const float rgba[4] = {0.30f, 0.30f, 0.35f, 1.0f};
      PushGeom(mjGEOM_SPHERE, size, pos, temp_mat, rgba);
    }

    // 简化指针实现：使用一个单一的长方体作为指针
    if (scene->ngeom < scene->maxgeom) {
      float half_len = pointer_total_len * 0.4f; // 长度减半，这样只向一个方向延伸
      float size[3] = {0.003f * gauge_scale, half_len, 0.003f * gauge_scale};
      // 计算指针延伸方向的单位向量（使用原始角度，不包含90度调整）
      float dir_x = cos(angle * 3.14159f / 180.0f);
      float dir_y = sin(angle * 3.14159f / 180.0f);
      // 沿着指针方向偏移一半长度，使指针从圆心向外延伸
      float pointer_pos[3];
      Place(pos, half_len * dir_x, half_len * dir_y, 0.0f, pointer_pos);
      const float rgba[4] = {1.0f, 0.10f, 0.10f, 1.0f};
      PushGeom(mjGEOM_BOX, size, pointer_pos, temp_mat, rgba);
    }

    // 指针尖端（增强视觉效果）
    if (scene->ngeom < scene->maxgeom) {
      float tip_size[3] = {0.005f * gauge_scale, 0.010f * gauge_scale, 0.005f * gauge_scale};
      // 指针尖端应该放在指针末端，而不是圆心
      float tip_pos[3];
      float tip_offset = pointer_total_len * 0.8f;
      float dir_x = cos(angle * 3.14159f / 180.0f);
      float dir_y = sin(angle * 3.14159f / 180.0f);
      Place(pos, tip_offset * dir_x, tip_offset * dir_y, 0.0f, tip_pos);
      const float tip_rgba[4] = {1.0f, 0.05f, 0.05f, 1.0f};
      PushGeom(mjGEOM_SPHERE, tip_size, tip_pos, temp_mat, tip_rgba);
    }
  }
  
    // 4. 中心固定点（小圆点）
  {
    float size[3] = {0.0075f * gauge_scale, 0.0075f * gauge_scale,
                     0.0075f * gauge_scale};
    float pos[3];
    Place(dashboard_pos, 0.0f, 0.0f, overlay_z + 0.0045f * gauge_scale, pos);
    const float rgba[4] = {0.95f, 0.18f, 0.20f, 1.0f};
    PushGeom(mjGEOM_SPHERE, size, pos, dashboard_rot_mat, rgba);
  }
  
  // 5. 数字速度显示（在仪表盘中央偏上）
  {
    float size[3] = {0.070f * gauge_scale, 0.026f * gauge_scale,
                     0.0040f * gauge_scale};
    float pos[3];
    Place(dashboard_pos, 0.0f, 0.018f, overlay_z + 0.0025f * gauge_scale, pos);
    const float rgba[4] = {0.00f, 0.00f, 0.00f, 0.70f};
    PushGeom(mjGEOM_BOX, size, pos, dashboard_rot_mat, rgba);
  }

  {
    char speed_label[50];
    std::snprintf(speed_label, sizeof(speed_label), "%.1f", speed_kmh_filtered);
    float pos[3];
    Place(dashboard_pos, 0.0f, 0.026f, overlay_z + 0.0010f * gauge_scale, pos);
    const float rgba[4] = {0.75f, 0.95f, 1.00f, 1.0f};
    PushLabel(pos, 0.08f, speed_label, rgba);
  }
  
  // 6. 添加"km/h"单位标签（在数字下方）
  {
    float pos[3];
    Place(dashboard_pos, 0.0f, 0.004f, overlay_z + 0.0010f * gauge_scale, pos);
    const float rgba[4] = {0.70f, 0.85f, 0.95f, 1.0f};
    PushLabel(pos, 0.05f, "km/h", rgba);
  }

  auto draw_small_gauge = [&](const float center[3], float ratio,
                              const float pointer_rgba[4],
                              const char* title_text,
                              const char* value_text) {
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;

    const float scale = gauge_scale * 0.70f;
    const float ring_radius = 0.10f * scale;
    const float ring_thickness = 0.015f * scale;
    const float pointer_half_len = 0.040f * scale;
    const float overlay_z = ring_thickness * 2.0f;

    if (scene->ngeom < scene->maxgeom) {
      float size[3] = {ring_radius * 1.14f, ring_radius * 1.14f,
                       ring_thickness * 1.35f};
      const float rgba[4] = {0.16f, 0.16f, 0.17f, 0.92f};
      PushGeom(mjGEOM_ELLIPSOID, size, center, dashboard_rot_mat, rgba);
    }

    if (scene->ngeom < scene->maxgeom) {
      float size[3] = {ring_radius * 1.06f, ring_radius * 1.06f,
                       ring_thickness * 1.10f};
      const float rgba[4] = {0.03f, 0.03f, 0.035f, 0.95f};
      PushGeom(mjGEOM_ELLIPSOID, size, center, dashboard_rot_mat, rgba);
    }

    if (scene->ngeom < scene->maxgeom) {
      float size[3] = {ring_radius * 1.14f, ring_radius * 1.14f,
                       ring_thickness * 1.55f};
      float pos[3];
      Place(center, 0.0f, 0.0f, 0.0015f * scale, pos);
      const float rgba[4] = {0.60f, 0.75f, 0.95f, 0.05f};
      PushGeom(mjGEOM_ELLIPSOID, size, pos, dashboard_rot_mat, rgba);
    }

    if (scene->ngeom < scene->maxgeom) {
      const int kSmallArcSegments = 18;
      for (int s = 0; s < kSmallArcSegments; s++) {
        if (scene->ngeom >= scene->maxgeom) break;
        const float seg_ratio =
            (kSmallArcSegments <= 1) ? 0.0f : static_cast<float>(s) / static_cast<float>(kSmallArcSegments - 1);
        float angle = 180.0f - 180.0f * seg_ratio;
        float rad_angle = angle * 3.14159f / 180.0f;

        float r = 0.35f, g = 0.35f, b = 0.35f, a = 0.55f;
        if (seg_ratio >= 0.83f) {
          r = 0.95f; g = 0.15f; b = 0.20f; a = 0.90f;
        }

        const float rr = ring_radius * 0.93f;
        float seg_size[3] = {0.0036f * scale, 0.0065f * scale,
                             0.0026f * scale};
        float seg_pos[3];
        Place(center, -rr * cos(rad_angle), rr * sin(rad_angle), overlay_z, seg_pos);

        double seg_rot_angle = static_cast<double>(angle) - 90.0;
        double rad_seg_rot = seg_rot_angle * 3.14159 / 180.0;
        double cos_t = cos(rad_seg_rot);
        double sin_t = sin(rad_seg_rot);
        double seg_rot_mat[9] = {cos_t, -sin_t, 0, sin_t, cos_t, 0, 0, 0, 1};

        double seg_mat[9];
        mju_mulMatMat(seg_mat, dashboard_rot_mat, seg_rot_mat, 3, 3, 3);
        const float rgba[4] = {r, g, b, a};
        PushGeom(mjGEOM_BOX, seg_size, seg_pos, seg_mat, rgba);
      }
    }

    if (scene->ngeom < scene->maxgeom) {
      float angle = 270.0f - 180.0f * ratio; // 改为270度，使初始角度指向-90°
      double pointer_angle = static_cast<double>(angle) + 90.0; // 与速度表保持一致，使用+90.0
      double rad_pointer_angle = pointer_angle * 3.14159 / 180.0;
      double cos_p = cos(rad_pointer_angle);
      double sin_p = sin(rad_pointer_angle);
      double pointer_rot_mat[9] = {cos_p, -sin_p, 0, sin_p, cos_p, 0, 0, 0, 1};
      double temp_mat[9];
      mju_mulMatMat(temp_mat, dashboard_rot_mat, pointer_rot_mat, 3, 3, 3);

      // 指针底座（增强立体感）
      if (scene->ngeom < scene->maxgeom) {
        float size[3] = {0.006f * scale, 0.006f * scale, 0.0045f * scale};
        float pos[3];
        Place(center, 0.0f, 0.0f, overlay_z + 0.0040f * scale, pos);
        const float rgba[4] = {0.30f, 0.30f, 0.35f, 1.0f};
        PushGeom(mjGEOM_SPHERE, size, pos, temp_mat, rgba);
      }

      // 计算指针方向向量，使用原始angle（与速度表保持一致）
      float rad_angle = angle * 3.14159f / 180.0f;
      float pointer_dir[2] = {cos(rad_angle), sin(rad_angle)};

      // 指针底部（靠近圆心部分）
      if (scene->ngeom < scene->maxgeom) {
        float bottom_length = pointer_half_len * 0.30f;
        float size[3] = {0.012f * scale, bottom_length, 0.006f * scale};
        // 从圆心开始，沿着指针方向放置底部
        float bottom_pos[3];
        Place(center, 
              (bottom_length / 2.0f) * pointer_dir[0], 
              (bottom_length / 2.0f) * pointer_dir[1], 
              overlay_z + 0.0040f * scale, 
              bottom_pos);
        float rgba[4] = {pointer_rgba[0], pointer_rgba[1], pointer_rgba[2], pointer_rgba[3]};
        PushGeom(mjGEOM_BOX, size, bottom_pos, temp_mat, rgba);
      }

      // 指针过渡部分
      if (scene->ngeom < scene->maxgeom) {
        float bottom_length = pointer_half_len * 0.30f;
        float transition_length = pointer_half_len * 0.35f;
        float size[3] = {0.006f * scale, transition_length, 0.005f * scale};
        // 从底部结束位置开始，沿着指针方向放置过渡部分
        float transition_pos[3];
        Place(center, 
              (bottom_length + transition_length / 2.0f) * pointer_dir[0], 
              (bottom_length + transition_length / 2.0f) * pointer_dir[1], 
              overlay_z + 0.0040f * scale, 
              transition_pos);
        float rgba[4] = {pointer_rgba[0], pointer_rgba[1], pointer_rgba[2], pointer_rgba[3]};
        PushGeom(mjGEOM_BOX, size, transition_pos, temp_mat, rgba);
      }

      // 指针顶部（远离圆心部分）
      if (scene->ngeom < scene->maxgeom) {
        float bottom_length = pointer_half_len * 0.30f;
        float transition_length = pointer_half_len * 0.35f;
        float top_length = pointer_half_len * 0.95f;
        float size[3] = {0.003f * scale, top_length, 0.003f * scale};
        // 从过渡部分结束位置开始，沿着指针方向放置顶部
        float top_pos[3];
        Place(center, 
              (bottom_length + transition_length + top_length / 2.0f) * pointer_dir[0], 
              (bottom_length + transition_length + top_length / 2.0f) * pointer_dir[1], 
              overlay_z + 0.0040f * scale, 
              top_pos);
        float rgba[4] = {pointer_rgba[0] * 1.1f, pointer_rgba[1] * 1.1f, pointer_rgba[2] * 1.1f, 1.0f};
        PushGeom(mjGEOM_BOX, size, top_pos, temp_mat, rgba);
      }

      // 指针尖端（增强视觉效果）
      if (scene->ngeom < scene->maxgeom) {
        float tip_pos[3];
        float tip_offset = pointer_half_len * 1.15f;
        // 使用与其他指针部分相同的pointer_dir计算尖端位置
        Place(center, 
              tip_offset * pointer_dir[0], 
              tip_offset * pointer_dir[1], 
              overlay_z + 0.0040f * scale, 
              tip_pos);
        float size[3] = {0.003f * scale, 0.003f * scale, 0.003f * scale};
        float rgba[4] = {pointer_rgba[0] * 1.2f, pointer_rgba[1] * 1.2f, pointer_rgba[2] * 1.2f, 1.0f};
        PushGeom(mjGEOM_SPHERE, size, tip_pos, temp_mat, rgba);
      }
    }

    // 中心固定点（小圆点）
    if (scene->ngeom < scene->maxgeom) {
      float size[3] = {0.006f * scale, 0.006f * scale, 0.006f * scale};
      float pos[3];
      Place(center, 0.0f, 0.0f, overlay_z + 0.0045f * scale, pos);
      const float rgba[4] = {0.95f, 0.18f, 0.20f, 1.0f};
      PushGeom(mjGEOM_SPHERE, size, pos, dashboard_rot_mat, rgba);
    }

    if (scene->ngeom < scene->maxgeom) {
      float pos[3];
      Place(center, 0.0f, 0.105f * scale, overlay_z + 0.0010f * scale, pos);
      const float rgba[4] = {0.92f, 0.92f, 0.92f, 1.0f};
      PushLabel(pos, 0.05f, title_text, rgba);
    }

    if (scene->ngeom < scene->maxgeom) {
      float pos[3];
      Place(center, 0.0f, -0.095f * scale, overlay_z + 0.0010f * scale, pos);
      const float rgba[4] = {0.92f, 0.92f, 0.92f, 1.0f};
      PushLabel(pos, 0.06f, value_text, rgba);
    }
  };

  auto draw_vertical_bar = [&](const float center[3], float ratio,
                               const float fill_rgba[4],
                               const char* title_text,
                               const char* value_text) {
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;

    const float scale = gauge_scale * 0.70f;
    const float bar_half_width = 0.030f * scale;
    const float bar_half_height = 0.095f * scale;
    const float bar_half_depth = 0.006f * scale;
    const float overlay_z = bar_half_depth * 2.0f;
    float front_center[3];
    Place(center, 0.0f, 0.0f, overlay_z, front_center);

    if (scene->ngeom < scene->maxgeom) {
      float size[3] = {bar_half_width, bar_half_height, bar_half_depth};
      const float rgba[4] = {0.06f, 0.06f, 0.07f, 0.80f};
      PushGeom(mjGEOM_BOX, size, front_center, dashboard_rot_mat, rgba);
    }

    if (ratio > 0.0f && scene->ngeom < scene->maxgeom) {
      const float fill_half_height = bar_half_height * ratio;
      const float y_offset = -bar_half_height + fill_half_height;

      // 从红色到绿色的平滑渐变，100%为绿色，0%为红色
      const float green_full[4] = {0.20f, 0.95f, 0.40f, 1.0f};  // 100%绿色
      const float red_empty[4] = {0.95f, 0.15f, 0.20f, 1.0f};  // 0%红色

      float fill_color[4];
      // 使用ratio直接作为插值因子，实现从红色到绿色的平滑过渡
      fill_color[0] = red_empty[0] + ratio * (green_full[0] - red_empty[0]);
      fill_color[1] = red_empty[1] + ratio * (green_full[1] - red_empty[1]);
      fill_color[2] = red_empty[2] + ratio * (green_full[2] - red_empty[2]);
      fill_color[3] = 1.0f;

      float size[3] = {bar_half_width * 0.92f, fill_half_height,
                       bar_half_depth * 2.0f};  // 显著增加深度，确保可见
      float pos[3];
      // 显著增加z轴偏移，确保填充条在背景条前面
      Place(center, 0.0f, y_offset, overlay_z + 0.003f * scale, pos);
      const float rgba[4] = {fill_color[0], fill_color[1], fill_color[2],
                             fill_color[3]};
      PushGeom(mjGEOM_BOX, size, pos, dashboard_rot_mat, rgba);
    }

    const float mark_half_height = 0.0036f * scale;
    const float mark_half_width = bar_half_width * 1.05f;
    const float mark_half_depth = 0.0024f * scale;
    for (int m = 0; m < 3; m++) {
      if (scene->ngeom >= scene->maxgeom) break;
      const float mark_ratio = (m == 0) ? 0.0f : ((m == 1) ? 0.5f : 1.0f);
      const float delta = -bar_half_height + (2.0f * bar_half_height * mark_ratio);
      float size[3] = {mark_half_width, mark_half_height, mark_half_depth};
      float pos[3] = {center[0] + delta * y_axis_world[0] + overlay_z * z_axis_world[0],
                      center[1] + delta * y_axis_world[1] + overlay_z * z_axis_world[1],
                      center[2] + delta * y_axis_world[2] + overlay_z * z_axis_world[2]};
      const float rgba[4] = {0.75f, 0.75f, 0.75f, 0.85f};
      PushGeom(mjGEOM_BOX, size, pos, dashboard_rot_mat, rgba);
    }

    if (scene->ngeom < scene->maxgeom) {
      float pos[3] = {center[0] + (bar_half_height + 0.012f * scale) * y_axis_world[0] + overlay_z * z_axis_world[0],
                      center[1] + (bar_half_height + 0.012f * scale) * y_axis_world[1] + overlay_z * z_axis_world[1],
                      center[2] + (bar_half_height + 0.012f * scale) * y_axis_world[2] + overlay_z * z_axis_world[2]};
      const float rgba[4] = {0.92f, 0.92f, 0.92f, 1.0f};
      PushLabel(pos, 0.045f, "F", rgba);
    }

    if (scene->ngeom < scene->maxgeom) {
      float pos[3] = {center[0] - (bar_half_height + 0.012f * scale) * y_axis_world[0] + overlay_z * z_axis_world[0],
                      center[1] - (bar_half_height + 0.012f * scale) * y_axis_world[1] + overlay_z * z_axis_world[1],
                      center[2] - (bar_half_height + 0.012f * scale) * y_axis_world[2] + overlay_z * z_axis_world[2]};
      const float rgba[4] = {0.92f, 0.92f, 0.92f, 1.0f};
      PushLabel(pos, 0.045f, "E", rgba);
    }

    const float title_offset = (bar_half_height + 0.055f * scale);
    if (scene->ngeom < scene->maxgeom) {
      float pos[3] = {center[0] + title_offset * y_axis_world[0] + overlay_z * z_axis_world[0],
                      center[1] + title_offset * y_axis_world[1] + overlay_z * z_axis_world[1],
                      center[2] + title_offset * y_axis_world[2] + overlay_z * z_axis_world[2]};
      const float rgba[4] = {0.85f, 0.85f, 0.85f, 1.0f};
      PushLabel(pos, 0.05f, title_text, rgba);
    }

    const float value_offset = (bar_half_height + 0.060f * scale);
    if (scene->ngeom < scene->maxgeom) {
      float pos[3] = {center[0] - value_offset * y_axis_world[0] + overlay_z * z_axis_world[0],
                      center[1] - value_offset * y_axis_world[1] + overlay_z * z_axis_world[1],
                      center[2] - value_offset * y_axis_world[2] + overlay_z * z_axis_world[2]};
      const float rgba[4] = {0.9f, 0.9f, 0.9f, 1.0f};
      PushLabel(pos, 0.06f, value_text, rgba);
    }
  };

  const float side_offset = bezel_radius + 0.12f * gauge_scale;

  float fuel_center[3] = {dashboard_pos[0] + side_offset * x_axis_world[0],
                          dashboard_pos[1] + side_offset * x_axis_world[1],
                          dashboard_pos[2] + side_offset * x_axis_world[2]};
  float rpm_center[3] = {dashboard_pos[0] - side_offset * x_axis_world[0],
                         dashboard_pos[1] - side_offset * x_axis_world[1],
                         dashboard_pos[2] - side_offset * x_axis_world[2]};



  char fuel_value_text[32];
  std::snprintf(fuel_value_text, sizeof(fuel_value_text), "%3.0f%%",
                fuel_percent);
  const float fuel_ratio = static_cast<float>(fuel_percent / 100.0);
  const float fuel_pointer_rgba[4] = {0.20f, 0.95f, 0.40f, 1.0f};
  draw_vertical_bar(fuel_center, fuel_ratio, fuel_pointer_rgba, "FUEL",
                    fuel_value_text);

  const double rpm_max = 600.0;
  float rpm_ratio_gauge = static_cast<float>(wheel_rpm / rpm_max);
  if (rpm_ratio_gauge < 0.0f) rpm_ratio_gauge = 0.0f;
  if (rpm_ratio_gauge > 1.0f) rpm_ratio_gauge = 1.0f;
  char rpm_value_text[32];
  std::snprintf(rpm_value_text, sizeof(rpm_value_text), "%.0f", wheel_rpm);
  const float rpm_pointer_rgba[4] = {0.95f, 0.30f, 0.90f, 1.0f};
  draw_small_gauge(rpm_center, rpm_ratio_gauge, rpm_pointer_rgba, "RPM",
                   rpm_value_text);



  const char* gear_text = "N";
  if (std::abs(throttle_filtered) > 0.12 || speed_ms > 0.05) {
    gear_text = (throttle_filtered >= 0.0) ? "D" : "R";
  }

  float gear_pos[3];
  Place(dashboard_pos, 0.060f * gauge_scale, 0.090f * gauge_scale,
        0.0040f * gauge_scale, gear_pos);

  if (scene->ngeom < scene->maxgeom) {
    float size[3] = {0.030f * gauge_scale, 0.020f * gauge_scale,
                     0.0032f * gauge_scale};
    const float rgba[4] = {0.00f, 0.00f, 0.00f, 0.70f};
    PushGeom(mjGEOM_BOX, size, gear_pos, dashboard_rot_mat, rgba);
  }

  {
    float pos[3];
    Place(gear_pos, 0.0f, 0.002f * gauge_scale, 0.0f, pos);
    float rgba[4];
    if (gear_text[0] == 'R') {
      rgba[0] = 0.98f;
      rgba[1] = 0.25f;
      rgba[2] = 0.25f;
      rgba[3] = 1.0f;
    } else if (gear_text[0] == 'D') {
      rgba[0] = 0.35f;
      rgba[1] = 0.95f;
      rgba[2] = 0.50f;
      rgba[3] = 1.0f;
    } else {
      rgba[0] = 0.90f;
      rgba[1] = 0.90f;
      rgba[2] = 0.90f;
      rgba[3] = 1.0f;
    }
    PushLabel(pos, 0.08f, gear_text, rgba);
  }

  if (fuel_ratio <= 0.15f && scene->ngeom < scene->maxgeom) {
    float pos[3];
    Place(fuel_center, 0.0f, 0.004f * gauge_scale, 0.0f, pos);
    const float rgba[4] = {0.98f, 0.25f, 0.25f, 1.0f};
    PushLabel(pos, 0.05f, "LOW", rgba);
  }
}

}  // namespace mjpc
