# Toe-vertical / parallel-shin candidate

Status: kinematic candidate, NOT dynamics-validated, NOT approved for hardware.
Source: upup1124/motions/gui00528900/motion_197_333.npz from the user-provided transfer.

This candidate changes the final one second using a quintic joint-space transition
and adds a one-second hold. It has 187 frames at 50 FPS. The original is unchanged.
Body transforms and velocities were recomputed, with a URDF-based floor-clearance
correction. See validation.json for measurements and limitations. The physical
feasibility of sustained toe support has NOT been demonstrated.

Not checked: full self-collision, USD collision geometry, dynamic balance, actuator
torque limits, closed-loop policy tracking. Do not deploy directly to a real robot.

Download motion.npz to this path in the project:
motions/gui00528900_toe_vertical_parallel_candidate/motion.npz

Kinematic preview (does not establish dynamic stability):
python scripts/replay_npz.py --robot oli --motion_file motions/gui00528900_toe_vertical_parallel_candidate/motion.npz
