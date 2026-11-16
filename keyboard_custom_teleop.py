import sys, time, select, termios, tty

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(port="/dev/ttyACM0", id="dave_arm", max_relative_target=5.0)
robot = SO101Follower(config)
robot.connect()

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
KEYMAP = {
    "q": ("shoulder_pan", +1),  "a": ("shoulder_pan", -1),
    "w": ("shoulder_lift", +1), "s": ("shoulder_lift", -1),
    "e": ("elbow_flex", +1),    "d": ("elbow_flex", -1),
    "r": ("wrist_flex", +1),    "f": ("wrist_flex", -1),
    "t": ("wrist_roll", +1),    "g": ("wrist_roll", -1),
    "y": ("gripper", +1),       "h": ("gripper", -1),
}
STEP = 1.0
HOLD = 0.12          # seconds a key counts as "down" after its last event
last_seen = {}       # key -> timestamp of last keypress event

obs = robot.get_observation()
target = {f"{j}.pos": obs[f"{j}.pos"] for j in JOINTS}

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)  # raw-ish keys, Ctrl-C still works
    while True:
        # drain all pending keystrokes
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x03":
                raise KeyboardInterrupt
            last_seen[ch] = time.time()

        now = time.time()
        for key, t in list(last_seen.items()):
            if now - t < HOLD and key in KEYMAP:
                j, d = KEYMAP[key]
                target[f"{j}.pos"] += d * STEP

        robot.send_action(target)
        time.sleep(1 / 30)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    robot.disconnect()