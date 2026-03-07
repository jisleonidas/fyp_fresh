# Algorithm Choices and Rationale

## 1. Stereo depth estimation

### Choice: Stereo matching with SGBM-class approach (reference implementation uses lightweight NumPy matcher)

Why:

- Good trade-off between compute cost and depth quality on embedded GPUs/CPUs
- More robust than local block matching in low-texture regions
- Mature and well-supported in OpenCV for hardware deployment
- Repository scaffold includes a pure-NumPy matcher for portability in constrained environments

Pipeline:

1. Rectify left/right images with calibration maps
2. Compute disparity (NumPy reference matcher or OpenCV StereoSGBM on target)
3. Optionally apply left-right consistency and filtering
4. Convert disparity to depth: $Z = \frac{fB}{d}$

Where:

- $Z$ = depth
- $f$ = focal length in pixels
- $B$ = baseline (m)
- $d$ = disparity (pixels)

Alternatives considered:

- Deep stereo networks (RAFT-Stereo, PSMNet): higher quality but heavier compute and deployment complexity
- StereoBM: faster but less accurate/noisy for outdoor scenes

## 2. Obstacle representation

### Choice: 2D local occupancy grid from depth projection

Why:

- Efficient for short-horizon navigation
- Integrates naturally with DWA local planning
- Works well for ground vehicles with limited pitch/roll

Method:

- Reproject depth points to vehicle frame
- Remove ground plane using RANSAC plane model
- Mark occupied cells with confidence accumulation
- Inflate occupied cells by safety radius

Alternatives:

- Full 3D voxel map: richer model but higher memory/compute
- Pure point-cloud reactive method: less stable for path optimization

## 3. Localization

### Choice: EKF fusion of wheel odometry + IMU (+ optional GPS)

Why:

- Lightweight and proven for buggy dynamics
- Reduces drift relative to odometry-only estimates
- Easy to tune and maintain

State (baseline):

- Position $(x, y)$
- Heading $\psi$
- Linear speed $v$
- Yaw rate $\dot{\psi}$

## 4. Global planner

### Choice: A* on occupancy/drivable grid

Why:

- Deterministic and interpretable
- Suitable for known/partially known map or slowly updated map
- Low implementation complexity and fast updates on moderate map sizes

Heuristic:

- Euclidean distance to goal with obstacle inflation costs in traversal function

## 5. Local planner

### Choice: Dynamic Window Approach (DWA)

Why:

- Real-time obstacle avoidance with vehicle dynamics constraints
- Produces feasible velocity commands directly
- Handles dynamic obstacles better than static geometric path-only approaches

Objective components:

- Progress toward local goal
- Obstacle clearance
- Heading alignment to global path
- Velocity preference (maintain forward progress)

## 6. Tracking controller

### Lateral: Pure Pursuit

- Stable and simple for Ackermann vehicles
- Easy lookahead tuning by speed

### Longitudinal: PID speed controller

- Tracks target speed from planner
- Enforces acceleration and jerk limits via command shaping

## 7. Safety and fallback logic

### Choice: Rule-based safety supervisor FSM

Checks:

- Time-to-collision threshold from occupancy and current velocity
- Sensor timeout/staleness
- Planner/control watchdog misses
- Actuator command saturation anomalies

Fallback policy:

1. Reduce speed to crawl in `DEGRADED`
2. Trigger full brake and neutral throttle in `EMERGENCY_STOP`
3. Require manual re-arm

## 8. Why this stack is appropriate for an FYP implementation

- Achievable on student timeline
- Interpretable behavior for demonstrations and reports
- Modular: each subsystem can later be replaced with ML-based components
