# MuJoCo MPC Simple Car 任务 - 作业报告

## 一、项目概述
### 1.1 作业背景
本作业基于MuJoCo物理引擎和MPC控制算法，实现了一个简单的汽车导航任务。通过在MuJoCo MPC框架中创建自定义任务，实现了差速驱动汽车的导航控制，使其能够追踪随机移动的目标位置。

### 1.2 实现目标
1. 掌握MuJoCo物理引擎的使用和大型开源项目的二次开发能力
2. 实现自定义汽车模型和物理属性设置
3. 定义导航任务和成本函数
4. 集成MPC控制算法实现目标追踪
5. 实现实时速度显示功能

### 1.3 开发环境
- 操作系统: Windows 11 + WSL2 Ubuntu 22.04
- 编译器: gcc 11.3.0
- CMake: 3.22.1
- 图形库: OpenGL 3.3+、GLFW、GLEW
- 物理引擎: MuJoCo 2.3.7
- 其他依赖: Eigen 3.4.0、OpenBLAS

## 二、技术方案
### 2.1 系统架构
本项目采用模块化设计，主要分为三个核心模块：

![](C:\Users\Davy\Desktop\39812fd552e10dce17e21b2087e265a8.png)

- **MuJoCo场景**: 定义车辆模型和物理环境，提供仿真数据
-  **数据提取模块**: 从MuJoCo的mjData结构中提取车辆状态数据
- **仪表盘渲染模块**: 使用OpenGL将仪表盘渲染为2D覆盖层

### 2.2 数据流程

![](C:\Users\Davy\Desktop\52093657fb75a707fadba060fdcdd27f.png) 

1. MuJoCo物理引擎运行仿真，更新车辆状态
2. 数据提取模块从mjData结构中获取速度、位置等信息
3. 将原始数据转换为仪表盘需要的格式（如m/s转换为km/h）
4. 渲染模块使用转换后的数据绘制仪表盘元素
5. 重复上述过程，实现实时更新

### 2.3 渲染方案
采用OpenGL 2D覆盖层（HUD）的渲染方案：
1. 使用正交投影模式，将3D场景渲染到窗口
2. 切换到2D正交投影模式，绘制仪表盘元素
3. 使用混合效果实现半透明背景
4. 禁用深度测试，确保仪表盘显示在最上层

## 三、实现细节
### 3.1 汽车模型创建
创建了汽车模型文件`car_model.xml`，包含以下关键特性：

```xml
<mujoco>
  <compiler autolimits="true"/>
  
  <option timestep="0.002" iterations="50" solver="Newton" tolerance="1e-10">
    <flag gravity="enable"/>
  </option>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <mesh name="chasis" scale=".01 .006 .0015"
      vertex=" 9   2   0
              -10  10  10
               9  -2   0
               10  3  -10
               10 -3  -10
              -8   10 -10
              -10 -10  10
              -8  -10 -10
              -5   0   20"/>
  </asset>

  <default>
    <joint damping=".05" armature="0.005"/>
    <geom friction="1 0.5 0.5" condim="3"/>
    <default class="wheel">
      <geom type="cylinder" size=".03 .01" rgba=".5 .5 1 1" friction="1.5 0.5 0.5"/>
    </default>
  </default>

  <worldbody>
    <geom type="plane" size="3 3 .01" material="grid" friction="1 0.5 0.5" condim="3"/>
    <body name="car" pos="0 0 .05">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.02 0.02 0.03"/>
      <geom name="chasis" type="mesh" mesh="chasis"/>
      <geom name="front wheel" pos=".08 0 -.015" type="sphere" size=".015" condim="1" priority="1"/>
      <body name="left wheel" pos="-.07 .06 0" zaxis="0 1 0">
        <joint name="left"/>
        <geom class="wheel"/>
      </body>
      <body name="right wheel" pos="-.07 -.06 0" zaxis="0 1 0">
        <joint name="right"/>
        <geom class="wheel"/>
      </body>
    </body>
    <body name="goal" pos="1.0 1.0 0.01" mocap="true">
      <geom type="sphere" size="0.1" rgba="0 1 0 0.5"/>
    </body>
  </worldbody>

  <tendon>
    <fixed name="forward">
      <joint joint="left" coef=".5"/>
      <joint joint="right" coef=".5"/>
    </fixed>
    <fixed name="turn">
      <joint joint="left" coef="-.5"/>
      <joint joint="right" coef=".5"/>
    </fixed>
  </tendon>

  <actuator>
    <motor name="forward" tendon="forward" ctrlrange="-1 1" gear="12"/>
    <motor name="turn" tendon="turn" ctrlrange="-1 1" gear="8"/>
  </actuator>
</mujoco>
```

### 3.2 任务实现
在`simple_car.cc`中实现了导航任务，包括残差计算和成本函数：

```cpp
namespace mjpc {

class SimpleCar : public Task {
 public:
  // 残差函数类
  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const SimpleCar* task) : BaseResidualFn(task) {}
    
    // 残差函数
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override {
      // 1. 位置残差（X方向）
      residual[0] = data->qpos[0] - data->mocap_pos[0];
      
      // 2. 位置残差（Y方向）
      residual[1] = data->qpos[1] - data->mocap_pos[1];
      
      // 3. 控制努力残差（前进）
      residual[2] = data->ctrl[0];
      
      // 4. 控制努力残差（转向）
      residual[3] = data->ctrl[1];
    }
  };

  // 目标位置更新函数
  void TransitionLocked(mjModel* model, mjData* data) override {
    // 汽车位置
    double car_pos[2] = {data->qpos[0], data->qpos[1]};
    
    // 目标位置
    double goal_pos[2] = {data->mocap_pos[0], data->mocap_pos[1]};
    
    // 计算距离
    double car_to_goal[2];
    mju_sub(car_to_goal, goal_pos, car_pos, 2);
    
    // 当到达目标时，随机更新目标位置
    if (mju_norm(car_to_goal, 2) < 0.2) {
      absl::BitGen gen_;
      data->mocap_pos[0] = absl::Uniform<double>(gen_, -2.0, 2.0);
      data->mocap_pos[1] = absl::Uniform<double>(gen_, -2.0, 2.0);
      data->mocap_pos[2] = 0.01;  // 保持在地面
    }
  }

  // 可视化函数
  void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const override {
    // 速度计算
    double speed_ms = 0.0;
    if (const double* car_velocity = SensorPtr("car_velocity")) {
      speed_ms = std::sqrt(car_velocity[0] * car_velocity[0] +
                           car_velocity[1] * car_velocity[1]);
    }
    double speed_kmh = speed_ms * 3.6;
    
    // 5分钟从100%降到0%的油量消耗逻辑
    const double total_time_minutes = 5.0;
    const double total_time_seconds = total_time_minutes * 60.0;
    double fuel_percent = 100.0 - (data->time / total_time_seconds) * 100.0;
    fuel_percent = clamp(fuel_percent, 0.0, 100.0);
    
    // ... 3D仪表盘渲染实现 ...
  }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc
```

### 3.3 速度显示功能
实现了实时速度显示功能，在`Visualize`函数中计算并显示汽车速度：

```cpp
// 在simple_car.cc的ModifyScene函数中
void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const override {
  // 速度计算
  double speed_ms = 0.0;
  if (const double* car_velocity = SensorPtr("car_velocity")) {
    speed_ms = std::sqrt(car_velocity[0] * car_velocity[0] +
                         car_velocity[1] * car_velocity[1]);
  }
  double speed_kmh = speed_ms * 3.6;
  
  // 5分钟从100%降到0%的油量消耗逻辑
  const double total_time_minutes = 5.0;
  const double total_time_seconds = total_time_minutes * 60.0;
  double fuel_percent = 100.0 - (data->time / total_time_seconds) * 100.0;
  fuel_percent = clamp(fuel_percent, 0.0, 100.0);
  
  // 实时打印速度和油量信息
  printf("\rCar Position: (%.2f, %.2f) | Speed: %.2f km/h | Fuel: %.1f%% ", 
         data->qpos[0], data->qpos[1], speed_kmh, fuel_percent);
  fflush(stdout);
  
  // 3D仪表盘渲染实现
  // ...
}
```

### 3.4 任务配置
在`task.xml`中配置了任务参数和成本权重：

```xml
<mujoco model="Simple Car Navigation">
  <include file="../common.xml"/>
  <include file="car_model.xml" />

  <size memory="1M" nconmax="500"/>

  <custom>
    <!-- agent -->
    <numeric name="agent_planner" data="1" />
    <numeric name="agent_horizon" data="2.0" />
    <numeric name="agent_timestep" data="0.02" />
    <numeric name="sampling_sample_width" data="0.02" />
    <numeric name="sampling_control_width" data="0.03" />
    <numeric name="sampling_spline_points" data="10" />
    <numeric name="sampling_exploration" data="0.5" />
    <numeric name="gradient_spline_points" data="10" />
    <numeric name="residual_Goal_Position_x" data="1.0 0.0 0.0 3.0" />
    <numeric name="residual_Goal_Position_y" data="1.0 0.0 0.0 3.0" />

    <!-- estimator -->
    <numeric name="estimator" data="0" />
  </custom>

  <sensor>
    <!-- cost - user sensors must be first and sequential -->
    <user name="Goal_Position_x" dim="1" user="0 10.0 0 100.0"/>
    <user name="Goal_Position_y" dim="1" user="0 10.0 0 100.0"/>
    <user name="Control_Forward" dim="1" user="0 0.1 0.0 1.0"/>
    <user name="Control_Turn" dim="1" user="0 0.1 0.0 1.0"/>

    <!-- speed/velocity sensors -->
    <framelinvel name="car_velocity" objtype="body" objname="car"/>
    <frameangvel name="car_angular_velocity" objtype="body" objname="car"/>
  </sensor>

  <worldbody>
    <body name="goal" mocap="true" pos="1 1 0.01">
      <geom name="goal" type="sphere" size="0.08" rgba="0 1 0 .5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
```

这些权重平衡了导航精度（位置残差）和控制努力（控制残差）。

## 四、遇到的问题和解决方案
### 问题1: 汽车模型稳定性问题
**现象**: 汽车在仿真过程中出现抖动或翻转
**原因**: 物理参数设置不当，如摩擦系数、阻尼和关节参数
**解决**: 调整物理参数以提高稳定性：
```xml
<option timestep="0.002" iterations="50" solver="Newton" tolerance="1e-6"/>
<default>
  <joint damping=".05" armature="0.005"/>
  <geom friction="1 0.5 0.5" condim="3"/>
  <default class="wheel">
    <geom type="cylinder" size=".03 .01" rgba=".5 .5 1 1" friction="1.5 0.5 0.5"/>
  </default>
</default>
```

### 问题2: MPC控制效果不佳
**现象**: 汽车难以准确到达目标位置或控制不稳定
**原因**: 成本函数权重设置不当，残差计算有误
**解决**: 调整成本函数权重和优化残差计算：
```xml
<task>
  <residual norm="L2">
    <weight index="0" value="1.0"/> <!-- 位置X权重 -->
    <weight index="1" value="1.0"/> <!-- 位置Y权重 -->
    <weight index="2" value="0.01"/> <!-- 控制前进权重 -->
    <weight index="3" value="0.01"/> <!-- 控制转向权重 -->
  </residual>
</task>
```

### 问题3: 目标位置更新逻辑问题
**现象**: 目标位置不移动或移动过于频繁
**原因**: 目标位置更新条件设置不当
**解决**: 实现合理的目标更新逻辑：
```cpp
// 在TransitionLocked函数中实现目标位置更新
void TransitionLocked(mjModel* model, mjData* data) override {
  // 汽车位置
  double car_pos[2] = {data->qpos[0], data->qpos[1]};
  
  // 目标位置
  double goal_pos[2] = {data->mocap_pos[0], data->mocap_pos[1]};
  
  // 计算距离
  double car_to_goal[2];
  mju_sub(car_to_goal, goal_pos, car_pos, 2);
  
  // 当到达目标时，随机更新目标位置
  if (mju_norm(car_to_goal, 2) < 0.2) {
    absl::BitGen gen_;
    data->mocap_pos[0] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[1] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[2] = 0.01;  // 保持在地面
  }
}
```

## 五、测试与结果
### 功能测试
1. **场景加载测试**: 成功加载`car_model.xml`场景和`SimpleCar`任务
2. **模型物理测试**: 汽车能够稳定行驶，没有出现抖动或翻转
3. **导航功能测试**: 汽车能够追踪目标位置，到达目标后目标自动移动
4. **速度显示测试**: 实时速度信息正确显示，单位转换准确（m/s和km/h）

### 性能测试
- 仿真帧率: 保持在60 FPS左右
- CPU使用率: 10-15%
- 内存占用: 约120 MB

### 控制性能
- **导航精度**: 汽车能够到达目标位置附近0.2单位内
- **响应时间**: 控制指令响应迅速，能够快速适应目标位置变化
- **最大速度**: 约1.5-2.0 m/s（5.4-7.2 km/h）

## 六、总结与展望
### 学习收获
1. 掌握了MuJoCo物理引擎的使用和大型开源项目的二次开发能力
2. 理解了MPC控制算法在机器人导航中的应用
3. 学习了如何创建自定义MuJoCo模型和任务
4. 掌握了如何实现残差函数和成本函数来定义导航任务

### 不足之处
1. 汽车模型相对简单，缺乏更复杂的物理特性
2. 导航策略较为基础，没有考虑避障等高级功能
3. 速度显示功能较为简单，仅在3D场景中显示文本

### 未来改进方向
1. 增强汽车模型的物理特性，如添加悬挂系统、摩擦力变化等
2. 实现更复杂的导航策略，包括避障、路径规划等
3. 优化控制算法参数，提高导航精度和稳定性
4. 添加更多的状态显示和可视化功能
5. 实现用户交互控制，允许手动操控汽车

## 参考资料
1. MuJoCo官方文档: https://mujoco.readthedocs.io/
2. Google DeepMind mujoco_mpc项目: https://github.com/google-deepmind/mujoco_mpc
3. 差速驱动机器人控制理论相关文献
4. MPC控制算法相关资料
5. C++编程与面向对象设计
