from ignite.handlers.clearml_logger import ClearMLLogger
import os
from clearml import Task
from omegaconf import DictConfig
import math


def setup_clearm(cfg : DictConfig):
    os.environ["CLEARML_API_HOST"] = f"http://{cfg.ip}:{8008}"
    os.environ["CLEARML_WEB_HOST"] = f"http://{cfg.ip}:{8080}"
    os.environ["CLEARML_FILES_HOST"] = f"http://{cfg.ip}:{8081}"
    os.environ["CLEARML_API_ACCESS_KEY"] = cfg.access_key
    os.environ["CLEARML_API_SECRET_KEY"] = cfg.secret_key

    task = Task.init(task_name=cfg.task_name, project_name=cfg.project_name,
                     auto_connect_arg_parser=False,
                     auto_connect_frameworks=False,
                     auto_resource_monitoring=False,
                     auto_connect_streams=False,
                     )

    clearml_logger = ClearMLLogger(task_name=cfg.task_name, project_name=cfg.project_name)
    return clearml_logger

def compute_action(
    dist,
    sin_theta,
    cos_theta,
    k_v=0.8,  # linear gain
    k_w=1.5,  # angular gain
    v_max=0.5,  # max linear velocity (m/s)
    w_max=1,
):  # max angular velocity (rad/s)
    """
    Compute linear and angular velocity commands for a differential drive robot
    given polar coordinates of the goal.

    Args:
        dist: distance to the goal (r >= 0)
        sin_theta: sin(angle_to_goal)
        cos_theta: cos(angle_to_goal)
        k_v: gain for linear velocity
        k_w: gain for angular velocity
        v_max: max linear speed
        w_max: max angular speed

    Returns:
        v, w: linear and angular velocities
    """
    theta = math.atan2(sin_theta, cos_theta)  # in [-pi, pi]
    w = k_w * theta

    v = k_v * dist

    v *= max(0.0, cos_theta)
    v = max(-v_max, min(v, v_max))
    w = max(-w_max, min(w, w_max))

    return v, w