from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)

config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="dave_arm",
)

robot = SO101Follower(config)

robot.connect(calibrate=False)

try:
    print("Robot connected!")
    observation = robot.get_observation()
    print("Current joint positions:")
    print(observation)
finally:
    robot.disconnect()
