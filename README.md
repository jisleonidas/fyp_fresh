# Stereo Vision Self-Driving Buggy

End-to-end baseline for an autonomous buggy using stereo vision depth perception, obstacle avoidance, and waypoint following.

## What this repository includes

- System architecture and interfaces
- Algorithm selections with rationale and trade-offs
- Modular Python implementation scaffold for:
  - Stereo depth estimation
  - Occupancy grid generation
  - Local obstacle avoidance
  - Global path planning
  - Trajectory tracking and drive control
  - Runtime supervisor and safety state machine
- Configuration and deployment specification files

## Top-level layout

- `docs/architecture.md` — full system architecture and dataflow
- `docs/algorithms.md` — chosen algorithms, why they were selected, alternatives
- `docs/system_spec.yaml` — machine-readable system specification
- `configs/default.yaml` — default runtime and tuning parameters
- `src/buggy_ai/` — modular software stack

## Mission profile (baseline)

- Platform: Ackermann steering buggy (can be adapted to differential drive)
- Environment: Semi-structured outdoor tracks with static obstacles and moderate lighting variation
- Sensor suite:
  - Stereo RGB cameras (left/right, synchronized)
  - IMU
  - Wheel encoder / odometry
  - Optional GPS (for large outdoor maps)

## Core pipeline

1. Acquire synchronized stereo images and motion data
2. Rectify and compute disparity
3. Convert disparity to depth and ground-plane-referenced occupancy grid
4. Fuse occupancy with short-horizon motion estimate
5. Run local planner (Dynamic Window Approach) with obstacle costs
6. Blend with global route (A* over grid map)
7. Track selected trajectory via pure pursuit + speed PID
8. Enforce safety supervisor (E-stop, degraded mode, fallback)

## Quick start (scaffold)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m buggy_ai.main --config configs/default.yaml
```

## Notes

- This is a production-oriented reference scaffold. Hardware-specific drivers and calibration files must be provided for your buggy.
- Safety interlocks should be implemented in dedicated hardware in addition to software checks.
