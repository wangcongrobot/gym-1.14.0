from gym.envs.robotics.fetch_env import FetchEnv
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv



from gym.envs.robotics.hand.reach import HandReachEnv
from gym.envs.robotics.hand.manipulate import HandBlockEnv
from gym.envs.robotics.hand.manipulate import HandEggEnv
from gym.envs.robotics.hand.manipulate import HandPenEnv

from gym.envs.robotics.hand.manipulate_touch_sensors import HandBlockTouchSensorsEnv
from gym.envs.robotics.hand.manipulate_touch_sensors import HandEggTouchSensorsEnv
from gym.envs.robotics.hand.manipulate_touch_sensors import HandPenTouchSensorsEnv

# add catcher3d env
from gym.envs.robotics.moving_target.catch import Catcher3dEnv

from gym.envs.robotics.fetch.catch import FetchCatchEnv

from gym.envs.robotics.ur5_gripper.ur5_catch import UR5CatchEnv
from gym.envs.robotics.ur5_gripper_env import UR5GripperEnv