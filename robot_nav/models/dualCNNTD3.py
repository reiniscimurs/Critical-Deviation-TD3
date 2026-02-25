import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf


class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()

        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.goal_embed = nn.Linear(3, 10)
        self.velocity_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.softsign = nn.Softsign()

    def forward(self, lidar, goal, velocity):
        if len(lidar.shape) == 1:
            lidar = lidar.unsqueeze(0)
            goal = goal.unsqueeze(0)
            velocity = velocity.unsqueeze(0)
        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))

        a = F.leaky_relu(self.velocity_embed(velocity))

        s = torch.concat((l, g, a), dim=-1)

        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.softsign(self.layer_3(s))
        return a

    def reward(self, batch_rewards, device):
        rewards = []
        for b in batch_rewards:
            rewards.append([b["collision"] + b["goal"] + b["step"] - b["deviation"]])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).reshape(
            -1, 1
        )
        return rewards


class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.goal_embed = nn.Linear(3, 10)
        self.velocity_embed = nn.Linear(2, 10)
        self.action_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, lidar, goal, velocity, action):
        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))
        v = F.leaky_relu(self.velocity_embed(velocity))
        a = F.leaky_relu(self.action_embed(action))

        s = torch.concat((l, g, v, a), dim=-1)

        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(action)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.leaky_relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(action)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.leaky_relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


class Dev_Actor(nn.Module):
    def __init__(self, action_dim):
        super(Dev_Actor, self).__init__()

        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.base_action_embed = nn.Linear(2, 10)
        self.velocity_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.softsign = nn.Softsign()

    def forward(self, lidar, velocity, base_action):
        if len(lidar.shape) == 1:
            lidar = lidar.unsqueeze(0)
        if len(velocity.shape) == 1:
            velocity = velocity.unsqueeze(0)
        if len(base_action.shape) == 1:
            base_action = base_action.unsqueeze(0)

        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        a = F.leaky_relu(self.base_action_embed(base_action))
        v = F.leaky_relu(self.velocity_embed(velocity))

        s = torch.concat((l, a, v), dim=-1)

        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.softsign(self.layer_3(s))
        return a

    def reward(self, batch_rewards, device):
        rewards = []
        for b in batch_rewards:
            rewards.append([b["collision"] - b["deviation"]])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).reshape(
            -1, 1
        )
        return rewards


class Dev_Critic(nn.Module):
    def __init__(self, action_dim):
        super(Dev_Critic, self).__init__()
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.velocity_embed = nn.Linear(2, 10)
        self.base_action_embed = nn.Linear(2, 10)
        self.action_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, lidar, velocity, base_action, action):
        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)
        v = F.leaky_relu(self.velocity_embed(velocity))
        b = F.leaky_relu(self.base_action_embed(base_action))
        a = F.leaky_relu(self.action_embed(action))

        s = torch.concat((l, v, b, a), dim=-1)

        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(action)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.leaky_relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(action)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.leaky_relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


class dualCNNTD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
    ):
        # Initialize the Actor network
        self.device = device
        self.actor = Actor(action_dim).to(self.device)
        self.actor_target = Actor(action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        self.dev_actor = Dev_Actor(action_dim).to(self.device)
        self.dev_actor_target = Dev_Actor(action_dim).to(self.device)
        self.dev_actor_target.load_state_dict(self.dev_actor.state_dict())
        self.dev_actor_optimizer = torch.optim.Adam(
            params=self.dev_actor.parameters(), lr=lr
        )

        # Initialize the Critic networks
        self.critic = Critic(action_dim).to(self.device)
        self.critic_target = Critic(action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.dev_critic = Dev_Critic(action_dim).to(self.device)
        self.dev_critic_target = Dev_Critic(action_dim).to(self.device)
        self.dev_critic_target.load_state_dict(self.dev_critic.state_dict())
        self.dev_critic_optimizer = torch.optim.Adam(
            params=self.dev_critic.parameters(), lr=lr
        )

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.iter_count = 0

    def get_action(self, obs, add_noise):
        if add_noise:
            action, dev_action = self.act(obs)
            action = (action + np.random.normal(0, 0.2, size=self.action_dim)).clip(
                -self.max_action, self.max_action
            )
            dev_action = (
                dev_action + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
            return action, dev_action
        else:
            return self.act(obs)

    def act(self, state):
        lidar = torch.tensor(state["lidar"], dtype=torch.float32, device=self.device)
        goal = torch.tensor(state["goal"], dtype=torch.float32, device=self.device)
        velocity = torch.tensor(
            state["velocity"], dtype=torch.float32, device=self.device
        )
        action = self.actor(lidar, goal, velocity)
        dev_action = (
            self.dev_actor(lidar, action, velocity).cpu().data.numpy().flatten()
        )
        return action.cpu().data.numpy().flatten(), dev_action

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            lidar = []
            goal = []
            velocity = []
            for b in batch_states:
                lidar.append(b["lidar"])
                goal.append(b["goal"])
                velocity.append(b["velocity"])

            lidar = torch.tensor(lidar, dtype=torch.float32, device=self.device)
            goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
            velocity = torch.tensor(velocity, dtype=torch.float32, device=self.device)

            next_lidar = []
            next_goal = []
            next_velocity = []
            for b in batch_next_states:
                next_lidar.append(b["lidar"])
                next_goal.append(b["goal"])
                next_velocity.append(b["velocity"])

            next_lidar = torch.tensor(
                next_lidar, dtype=torch.float32, device=self.device
            )
            next_goal = torch.tensor(next_goal, dtype=torch.float32, device=self.device)
            next_velocity = torch.tensor(
                next_velocity, dtype=torch.float32, device=self.device
            )

            action = torch.tensor(
                batch_actions, dtype=torch.float32, device=self.device
            )
            # reward = torch.Tensor(batch_rewards).to(self.device).reshape(-1, 1)
            done = torch.tensor(
                batch_dones, dtype=torch.float32, device=self.device
            ).reshape(-1, 1)

            dev_action = action[..., 2:]
            base_action = action[..., :2]
            # ------------------------------
            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_lidar, next_goal, next_velocity)

            # Add noise to the action
            noise = torch.randn_like(next_action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            # next_full_action = (next_action + next_override + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(
                next_lidar, next_goal, next_velocity, next_action
            )

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation

            with torch.no_grad():
                target_Q = self.actor.reward(batch_rewards, self.device) + (
                    (1 - done) * discount * target_Q
                )

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(lidar, goal, velocity, base_action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            # ------------------------------

            dev_next_action = self.dev_actor_target(
                next_lidar, next_velocity, next_action.detach()
            )
            noise = torch.randn_like(dev_next_action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            dev_next_action = (dev_next_action + noise).clamp(
                -self.max_action, self.max_action
            )

            dev_target_Q1, dev_target_Q2 = self.dev_critic_target(
                next_lidar, next_velocity, next_action.detach(), dev_next_action
            )
            dev_target_Q = torch.min(dev_target_Q1, dev_target_Q2)
            with torch.no_grad():
                dev_target_Q = self.dev_actor.reward(batch_rewards, self.device) + (
                    (1 - done) * discount * dev_target_Q
                )
            dev_current_Q1, dev_current_Q2 = self.dev_critic(
                lidar, velocity, base_action, dev_action
            )

            dev_critic_loss = F.mse_loss(dev_current_Q1, dev_target_Q) + F.mse_loss(
                dev_current_Q2, dev_target_Q
            )

            self.dev_critic_optimizer.zero_grad()
            dev_critic_loss.backward()
            self.dev_critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                self.critic.eval()
                self.dev_critic.eval()
                actor_grad, _ = self.critic(
                    lidar, goal, velocity, self.actor(lidar, goal, velocity)
                )
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                dev_actor_grad, _ = self.dev_critic(
                    lidar,
                    velocity,
                    base_action,
                    self.dev_actor(lidar, velocity, base_action),
                )
                dev_actor_grad = -dev_actor_grad.mean()
                self.dev_actor_optimizer.zero_grad()
                dev_actor_grad.backward()
                self.dev_actor_optimizer.step()

                self.critic.train()
                self.dev_critic.train()

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.dev_actor.parameters(), self.dev_actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.dev_critic.parameters(), self.dev_critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        return {
            "loss": av_loss / iterations,
            "avg_Q": av_Q / iterations,
            "max_Q": max_Q,
        }

    def prepare_state(self, observation):
        latest_scan = observation["latest_scan"]
        distance = observation["distance"]
        cos = observation["cos"]
        sin = observation["sin"]
        collision = observation["collision"]
        goal = observation["goal"]
        action = observation["action"]
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0
        latest_scan /= 7

        # Normalize to [0, 1] range
        distance /= 10
        lin_vel = action[0] * 2
        ang_vel = (action[1] + 1) / 2
        # state = latest_scan.tolist() + [lin_vel, ang_vel]

        # assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0
        state = {
            "lidar": latest_scan.tolist(),
            "goal": [distance, cos, sin],
            "velocity": [lin_vel, ang_vel],
        }

        return state, terminal
