from sim.dev_sim import DEV_SIM


class DUAL_SIM(DEV_SIM):

    @staticmethod
    def get_reward(goal, collision, action, override_action, laser_scan, switch):

        reward = {
            "goal": 0.0,
            "collision": 0.0,
            "step": 0.0,
            "deviation": 0.0,
        }
        if goal:
            reward["goal"] = 100.0
        if collision:
            reward["collision"] = -100.0
        reward["step"] = action[0] - abs(action[1]) / 2
        reward["deviation"] = (override_action[0] ** 2) + (override_action[1] ** 2)

        return reward
