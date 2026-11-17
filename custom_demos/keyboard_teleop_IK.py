import os, sys, time, select, termios, tty
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.model.kinematics import RobotKinematics  # pip install placo

URDF = "/home/colin/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# use_degrees=True so joint values match the URDF's units
config = SO101FollowerConfig(port="/dev/ttyACM0", id="dave_arm",
                             max_relative_target=5.0, use_degrees=True)
robot = SO101Follower(config)
robot.connect()

IK_Joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]      # IK-controlled
LOCKED_Joints = ["wrist_roll"]                                   # locked
kin = RobotKinematics(urdf_path=URDF, target_frame_name="gripper_frame_link",
                      joint_names=IK_Joints + LOCKED_Joints)

# ---- locked pose: set explicit values here, or leave None to lock wherever it is at startup
LOCKED = {"wrist_roll": 0, "gripper": 50}
START = {"wrist_flex": 0}   # joints to drive to a start pose

obs = robot.get_observation()
for k in LOCKED:
    if LOCKED[k] is None:
        LOCKED[k] = obs[f"{k}.pos"]
 
TARGETS = {**LOCKED, **START} # combines LOCKED and START dicts into new dict
# Move to the locked pose first (max_relative_target limits each send to 5 deg)
for _ in range(200):
    robot.send_action({**{f"{j}.pos": obs[f"{j}.pos"] for j in IK_Joints},
                       **{f"{k}.pos": v for k, v in TARGETS.items()}})
    obs = robot.get_observation()
    if all(abs(obs[f"{k}.pos"] - v) < 1.0 for k, v in TARGETS.items()):
        break
    time.sleep(1 / 30)

q = np.array([obs[f"{j}.pos"] for j in IK_Joints], dtype=float)   # degrees
wrist = np.array([LOCKED[j] for j in LOCKED_Joints], dtype=float)

def fk(q3):
    return kin.forward_kinematics(np.concatenate([q3, [q[3]], wrist]))[:3, 3]

def jacobian(q3, eps_deg=0.5):
    """3x3 numeric Jacobian, meters per radian, wrist held fixed."""
    base = fk(q3)
    J = np.zeros((3, 3))
    for i in range(3):
        dq = q3.copy(); dq[i] += eps_deg
        J[:, i] = (fk(dq) - base) / np.radians(eps_deg)
    return J

def ik_step(q3, dx, lam=0.02, max_deg=3.0):
    """Damped least-squares: joint delta (deg) that moves the tip by dx (m)."""
    J = jacobian(q3)
    dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), dx)
    return np.clip(np.degrees(dq), -max_deg, max_deg)

# Directions in the robot base frame (x forward, y left, z up).
# If any axis feels backwards, flip its sign here.
UP, DOWN, LEFT, RIGHT = "\x1b[A", "\x1b[B", "\x1b[D", "\x1b[C"
KEYMAP = {
    UP:    np.array([0, 0, +1.0]),
    DOWN:  np.array([0, 0, -1.0]),
    LEFT:  np.array([0, +1.0, 0]),
    RIGHT: np.array([0, -1.0, 0]),
    "w":   np.array([+1.0, 0, 0]),
    "s":   np.array([-1.0, 0, 0]),
}
SPEED = 0.05         # m/s tip speed
HZ = 30
HOLD = 0.12          # seconds a key counts as "down" after its last event
last_seen = {}

def read_keys():
    """Return list of keys from pending stdin, with arrow escape sequences intact."""
    keys = []
    while select.select([sys.stdin], [], [], 0)[0]:
        buf = os.read(fd, 64).decode(errors="ignore")
        i = 0
        while i < len(buf):
            if buf.startswith("\x1b[", i) and i + 2 < len(buf):
                keys.append(buf[i:i + 3]); i += 3
            else:
                keys.append(buf[i]); i += 1
    return keys

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while True:
        for k in read_keys():
            if k == "\x03":
                raise KeyboardInterrupt
            last_seen[k] = time.time()

        now = time.time()
        v = np.zeros(3)
        for key, t in last_seen.items():
            if now - t < HOLD and key in KEYMAP:
                v += KEYMAP[key]

        if np.any(v):
            dx = v / np.linalg.norm(v) * SPEED / HZ
            q[:3] += ik_step(q, dx)
        action = {f"{j}.pos": float(q[i]) for i, j in enumerate(IK_Joints)}
        action.update({f"{k}.pos": v_ for k, v_ in LOCKED.items()})
        robot.send_action(action)
        time.sleep(1 / HZ)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    robot.disconnect()