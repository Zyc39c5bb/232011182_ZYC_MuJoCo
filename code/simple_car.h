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

#ifndef MJPC_TASKS_SIMPLE_CAR_SIMPLE_CAR_H_
#define MJPC_TASKS_SIMPLE_CAR_SIMPLE_CAR_H_

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class SimpleCar : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const SimpleCar* task) : BaseResidualFn(task), prev_ctrl_forward_(0.0), prev_ctrl_turn_(0.0) {}
    // ------- Residuals for simple_car task ------
    //   Number of residuals: 6
    //     Residual (0-1): position - goal_position (x, y)
    //     Residual (2): control - forward
    //     Residual (3): control - turn
    //     Residual (4): control change rate - forward
    //     Residual (5): control change rate - turn
    // ------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
   
   private:
    mutable double prev_ctrl_forward_;  // previous forward control
    mutable double prev_ctrl_turn_;  // previous turn control
  };

  SimpleCar() : residual_(this), reached_goal_time_(-1.0), stay_duration_(2.0), fuel_percent_(100.0) {
    visualize = 1;  // enable visualization
  }

  // -------- Transition for simple_car task --------
  //   If car is within tolerance of goal ->
  //   move goal randomly after a stay duration.
  // ------------------------------------------------
  void TransitionLocked(mjModel* model, mjData* data) override;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
 ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  double reached_goal_time_;  // time when car reached goal
  const double stay_duration_;  // duration to stay at goal before moving
  mutable double fuel_percent_;  // current fuel percentage (mutable for ModifyScene)
};
}  // namespace mjpc

#endif  // MJPC_TASKS_SIMPLE_CAR_SIMPLE_CAR_H_
