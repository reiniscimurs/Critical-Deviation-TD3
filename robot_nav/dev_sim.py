import numpy as np

from sim import SIM


class DEV_SIM(SIM):

    def step(
        self,
        lin_velocity=0.0,
        ang_velocity=0.1,
        override_lin=-0.0,
        override_ang=-0.1,
        switch=0,
    ):
        action = [(lin_velocity + 1) / 4, ang_velocity]
        override_action = [override_lin / 2, override_ang * 2]
        if switch:
            a_in = action[:]
        else:
            a_in = [
                action[0] + override_action[0],
                action[1] + override_action[1],
            ]
        self.env.step(action_id=0, action=np.array([[a_in[0]], [a_in[1]]]))
        self.env.render()

        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        robot_state = self.env.get_robot_state()
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        goal = self.env.robot.arrive
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        reward = self.get_reward(
            goal, collision, action, override_action, latest_scan, switch
        )
        observation = {
            "latest_scan": latest_scan,
            "distance": distance,
            "cos": cos,
            "sin": sin,
            "collision": collision,
            "goal": goal,
            "action": a_in,
            "reward": reward,
        }
        return observation

    @staticmethod
    def get_reward(goal, collision, action, override_action, laser_scan, switch):
        if switch:
            if goal:
                return 100.0
            elif collision:
                return -100.0
            else:
                deviation = (override_action[0] ** 2) + (override_action[1] ** 2)
                return action[0] - abs(action[1]) / 2 - deviation

        else:
            if collision:
                return -100.0
            else:
                deviation = (override_action[0] ** 2) + (override_action[1] ** 2)
                return -deviation
