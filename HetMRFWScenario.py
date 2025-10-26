import torch
from torch import Tensor

from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic  # Multirotor
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle  # Fixed-Wing
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import (
    Color,
    ScenarioUtils,
)


class HetMRFWScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False

        # agents
        self.n_multirotor = kwargs.pop("multirotor", 4)
        self.n_fixedwing = kwargs.pop("fixedwing", 2)
        self.n_agents = self.n_multirotor + self.n_fixedwing

        # rendering
        self.render_action = kwargs.pop("render_action", False)
        self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1)
        self.shared_rew = kwargs.pop("shared_rew", False)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        # obstacles
        self.n_obstacles = kwargs.pop("n_obstacles", 0)

        # agent attr and sensors
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        view_area = (self.world_spawning_x * self.world_spawning_y) / 4
        self.lidar_range = kwargs.pop("lidar_range", ((view_area) / torch.pi) ** (1 / 2))
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 8)
        self.min_distance_between_entities = kwargs.pop("min_distance_between_entities", self.agent_radius * 2 + 0.05)
        self.min_collision_distance = kwargs.pop("min_collision_distance", 0.005)

        # check if kwargs have all been used
        ScenarioUtils.check_kwargs_consumed(kwargs)

        ### Review of kwargs ###
        # n_multirotor
        # n_fixedwing
        # render_action
        # world_spawning_x
        # world_spawning_y
        # shared_rew
        # final_reward
        # n_obstacles
        # agent_radius
        # lidar_range
        # n_lidar_rays
        # min_distance_between_entities

        world = World(batch_dim=batch_dim, device=device)

        self.goals = []
        # Add agents and landmarks to environment
        for i in range(self.n_agents):
            if (i < self.n_multirotor):
                agent = Agent(
                    name=f"multirotor_{i}",
                    collide=True,
                    shape=Sphere(radius=self.agent_radius),
                    color=Color.RED,
                    dynamics=Holonomic(),
                    render_action=self.render_action,
                    u_range=[1, 1],
                    sensors=[
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=lambda e: isinstance(
                                e, Agent
                            ),
                            angle_start=0.0,
                            angle_end=2 * torch.pi,
                        )
                    ]  # LIDAR sensor TODO: Change this to None
                )
            else:
                max_steering_angle = torch.pi / 4
                sensors = [
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.lidar_range,
                        entity_filter=lambda e: isinstance(
                            e, Agent
                        ),
                        angle_start=0.0,
                        angle_end=2 * torch.pi,
                    )
                ]  # LIDAR sensor

                agent = Agent(
                    name=f"fixedwing_{i - self.n_multirotor}",
                    collide=True,
                    shape=Box(length=self.agent_radius * 2, width=self.agent_radius),
                    color=Color.ORANGE,
                    sensors=sensors,
                    u_range=[1, max_steering_angle],
                    render_action=self.render_action,
                    dynamics=KinematicBicycle(
                        world,
                        width=self.agent_radius,
                        l_f=self.agent_radius,
                        l_r=self.agent_radius,
                        max_steering_angle=max_steering_angle
                    )
                )
            agent.pos_rew = torch.zeros(
                batch_dim, device=device
            )
            agent.agent_collision_rew = (
                agent.pos_rew.clone()
            )
            world.add_agent(agent)
            # add landmarks
            goal = Landmark(
                name=f"landmark_{i}",
                collide=False,
                color=Color.GREEN
            )
            world.add_landmark(goal)
            self.goals.append(goal)
            agent.goal = goal

        self.pos_rew = torch.zeros(
            batch_dim, device=device
        )  # global position reward
        self.final_rew = (
            self.pos_rew.clone()
        )  # global done reward
        self.all_goal_reached = (
            self.pos_rew.clone()
        )  # if all goals have been reached

        # (optional) add obstacles
        self.obstacles = []  # store obstacles here for convenience
        for k in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{k}",
                collide=True,
                color=Color.BLACK,
                shape=Sphere(radius=self.agent_radius * 2 / 3),
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents
            + self.goals
            + self.obstacles,
            self.world,
            env_index,
            self.min_distance_between_entities,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y)
        )

        for agent in self.world.agents:
            if env_index is None:
                agent.goal_dist = torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )  # distance of the agent to the goal, use it to compute the reward
            else:
                agent.goal_dist[env_index] = torch.linalg.vector_norm(
                    agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            # resetting prev global rews
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                a.agent_collision_rew[:] = 0
                distance_to_goal = torch.linalg.vector_norm(
                    a.state.pos - a.goal.state.pos,
                    dim=-1
                )
                a.on_goal = distance_to_goal < a.shape.circumscribed_radius()

                a.pos_rew = a.goal_dist - distance_to_goal

                a.goal_dist = distance_to_goal

                self.pos_rew += a.pos_rew

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1
            )
            self.final_rew[self.all_goal_reached] = self.final_reward
            for i, a in enumerate(self.world.agents):
                # agent collision
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[distance <= self.min_collision_distance
                                              ] += -1
                        b.agent_collision_rew[distance <= self.min_collision_distance
                                              ] += -1
                for b in self.obstacles:
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                            ] += self.agent_collision_penalty
        pos_reward = (
            self.pos_rew if self.shared_rew else agent.pos_rew
        )  # Choose global or local reward based on configuration
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def observation(self, agent: Agent):
        obs = {
            "obs": torch.cat(
                [
                    agent.state.pos - agent.goal.state.pos
                ]  # Relative position to goal (fundamental)
                + [
                    agent.state.pos - obstacle.state.pos for obstacle in self.obstacles
                ]  # Relative position to obstacles (fundamental)
                + [
                    sensor._max_range - sensor.measure() for sensor in agent.sensors
                ],  # LIDAR to avoid other agents
                dim=-1,
            ),
            "pos": agent.state.pos,
            "vel": agent.state.vel,
        }
        if isinstance(agent.dynamics, KinematicBicycle):
            # Non hoonomic agents need to know angular states
            obs.update(
                {
                    "rot": agent.state.rot,
                    "ang_vel": agent.state.ang_vel,
                }
            )
        return obs

    def fixedwing_reward(self):
        pass

    def multirotor_reward(self):
        pass