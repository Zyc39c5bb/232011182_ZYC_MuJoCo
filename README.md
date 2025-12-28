# MuJoCo MPC 汽车仪表盘项目

## 项目信息
- **学号**: 232011182
- **姓名**: 赵奕宸
- **班级**: 计科2305班
- **完成日期**: 2025年12月28日

## 项目概述
本项目基于MuJoCo物理引擎和MPC（模型预测控制）算法，实现了一个简单的汽车导航任务。通过在MuJoCo MPC框架中创建自定义任务，实现了差速驱动汽车的导航控制，使其能够追踪随机移动的目标位置，实现了速度表、转速表和油量显示等功能。

主要实现内容包括：
1. 自定义汽车模型和物理属性设置
2. 基于MPC的导航控制算法集成
3. 实时速度和油量显示功能
4. 目标位置自动更新逻辑

## 环境要求
- 操作系统: Windows 10/11 (WSL2 Ubuntu 22.04)
- 编译器: gcc 11.3.0
- CMake: 3.22.1
- 依赖库: MuJoCo 2.3.7 、Eigen 3.4.0 、absl库

## 编译和运行

### 编译步骤
```bash
cd mujoco_projects/mujoco_mpc/
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

### 运行
```bash
./bin/mjpc --task SimpleCar
```

## 功能说明

### 已实现功能
- [√] 速度表显示（0-10 km/h）
- [√] 转速表显示（0-1000 RPM）
- [√] 油量显示（进度条形式）
- [√] 实时数据更新

### 进阶功能
- [√] UI动画效果（指针平滑移动）
- [√] 油量警告提示（过低时显示红色）

## 文件说明

-`mjpc/tasks/simple_car/simple_car.h`: 定义了SimpleCar任务的类结构和接口
-`mjpc/tasks/simple_car/simple_car.cc`: 实现SimpleCar任务的核心逻辑
-`mjpc/tasks/simple_car/car_model.xml`: 定义汽车的几何结构和物理属性
-`mjpc/tasks/simple_car/task.xml`: 配置任务参数和MPC控制参数

## 已知问题
- 暂无

## 参考资料
- MuJoCo官方文档: https://mujoco.readthedocs.io/
- OpenGL教程: https://learnopengl-cn.github.io/
- Google DeepMind mujoco_mpc项目: https://github.com/google-deepmind/mujoco_mpc
