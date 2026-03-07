# System Architecture Specification

## 1. Objectives

The autonomous buggy shall:

1. Detect obstacles in near real-time using stereo depth
2. Avoid collisions using local reactive planning
3. Follow mission waypoints using global planning and local control
4. Fail safely under perception, planning, or actuation faults

Target performance (baseline):

- Perception update: >= 15 Hz
- Planning/control loop: >= 20 Hz
- Emergency stop trigger latency: <= 100 ms software path

## 2. Layered architecture

### Sensing Layer

- Stereo camera driver (timestamped left/right frames)
- IMU stream
- Wheel odometry
- Time synchronization service

### Perception Layer

- Camera calibration + rectification
- Stereo matching (Semi-Global Block Matching)
- Depth reconstruction
- Ground-plane estimation and obstacle segmentation
- Local occupancy grid builder

### Localization Layer

- Short-horizon state estimation from odometry + IMU
- Optional GPS fusion for large-area global positioning

### Planning Layer

- Global planner: A* over static/drivable grid graph
- Local planner: Dynamic Window Approach (DWA)
- Costmap fusion: obstacle inflation + traversability + goal alignment

### Control Layer

- Lateral controller: Pure Pursuit
- Longitudinal controller: PID speed tracking
- Actuator output: steering angle + throttle/brake

### Safety Layer

- Safety supervisor finite-state machine:
  - `INIT`
  - `READY`
  - `AUTONOMOUS`
  - `DEGRADED`
  - `EMERGENCY_STOP`
- Fault detection on sensor staleness, planner timeout, control saturation

## 3. Runtime components and interfaces

| Component | Inputs | Outputs | Nominal Rate |
|---|---|---|---|
| StereoProcessor | Left/Right frames, calibration | Disparity, depth map | 15–30 Hz |
| OccupancyMapper | Depth map, pose | Local occupancy grid | 15–20 Hz |
| LocalPlanner | Occupancy, pose, velocity, local goal | Trajectory candidates + selected command | 20 Hz |
| GlobalPlanner | Map, start, goal | Waypoint path | 1–2 Hz or on demand |
| Controller | Selected trajectory, state | steering, throttle, brake | 20–50 Hz |
| SafetySupervisor | Health metrics + commands | gated command / E-stop | 50 Hz |

## 4. Dataflow

1. Frames are synchronized and rectified.
2. Stereo matcher generates disparity.
3. Disparity is converted to depth using baseline and focal length.
4. Depth is projected into a local occupancy grid.
5. Planner computes safe motion command with global path context.
6. Controller tracks command.
7. Safety supervisor validates all outputs and can override to safe stop.

## 5. State machine behavior

- `INIT` -> `READY`: all critical sensors healthy and calibrated.
- `READY` -> `AUTONOMOUS`: mission armed and route available.
- `AUTONOMOUS` -> `DEGRADED`: partial sensor loss but safe crawl possible.
- Any state -> `EMERGENCY_STOP`: collision imminent, stale control, or hardware E-stop.
- `EMERGENCY_STOP` -> `READY`: manual reset and health checks pass.

## 6. Deployment architecture

Recommended compute split:

- Onboard compute unit (NVIDIA Jetson / x86): perception + planning + control
- Microcontroller (real-time): motor control loops + hardware safety interlock

Network:

- ROS2 DDS over local Ethernet/CAN bridge for telemetry and diagnostics
- Isolate safety-critical stop lines from high-level software stack

## 7. Safety and verification requirements

- Redundant stop path (software + hardware E-stop)
- Watchdog for planner and controller heartbeat
- Sensor freshness window checks
- Regression scenarios:
  - Static obstacle in lane
  - Sudden obstacle appearance
  - Perception degradation (low light / texture-poor surfaces)
  - Sharp curve tracking under speed limits
