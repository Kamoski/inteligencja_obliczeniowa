import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt


def evaluate_model(model, env, num_episodes=100):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        print(f"Current episode: {i}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

def find_best_route(model, env):
    obs = env.reset()
    done = False
    route = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        route.append(action)
    return route

def plot_route(route, env):
    points = env.points
    route_points = [points[action] for action in route]

    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=200, c='red', marker='o', label='Locations')
    plt.scatter(points[0, 0], points[0, 1], s=200, c='blue', marker='s', label='Depot')

    for i, txt in enumerate(range(env.num_points)):
        plt.annotate(txt, (points[i, 0] + 1, points[i, 1] + 1))

    route_points = [points[0]] + route_points
    route_lines = np.array(route_points)
    plt.plot(route_lines[:, 0], route_lines[:, 1], 'b--', label='Route')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.title('Vehicle Routing')
    plt.show()

class VehicleRoutingEnv(Env):
    def __init__(self):
        self.num_points = 10
        self.map_range = (0, 100)
        self.vehicle_capacity = 1000
        self.min_vehicles = 3
        self.max_vehicles = 6
        self.goods_range = (100, 200)

        self.action_space = Discrete(self.num_points)
        self.observation_space = Box(
            low=0,
            high=self.vehicle_capacity,
            shape=(2 * self.num_points + self.num_points,)
        )
        self.reset()

    def reset(self):
        self.points = np.random.uniform(low=self.map_range[0], high=self.map_range[1], size=(self.num_points, 2))
        self.goods = np.random.uniform(low=self.goods_range[0], high=self.goods_range[1], size=(self.num_points,))
        self.remaining_goods = self.goods.copy()
        self.current_vehicle = 0
        self.num_vehicles = random.randint(self.min_vehicles, self.max_vehicles)
        self.vehicle_positions = np.tile(self.points[0], (self.num_vehicles, 1))
        self.vehicle_loads = np.full(self.num_vehicles, self.vehicle_capacity)
        self.visited_points = {i: [] for i in range(self.num_vehicles)}

        return self._get_observation()

    def step(self, action):
        epsilon = 0.1  # Probability of selecting a random action
        if np.random.rand() < epsilon:
            unvisited_points = [i for i in range(1, self.num_points) if i not in self.visited_points[self.current_vehicle]]
            if unvisited_points:
                action = np.random.choice(unvisited_points)
            else:
                action = 0
        target = action
        if target == 0 and self.remaining_goods[0] > 0:
            distance = np.linalg.norm(self.vehicle_positions[self.current_vehicle] - self.points[target])
            reward = -distance

        # Check if the target has already been visited by the current vehicle
        if target in self.visited_points[self.current_vehicle] or self.remaining_goods[target] == 0:
            reward = -100  # large penalty for trying to visit a visited point or a point with no remaining goods
            return self._get_observation(), reward, False, {}

        distance = np.linalg.norm(self.vehicle_positions[self.current_vehicle] - self.points[target])
        reward = -distance

        load = min(self.remaining_goods[target], self.vehicle_loads[self.current_vehicle])
        if load > 0:
            reward += 5  # small positive reward for delivering goods

        self.remaining_goods[target] -= load
        self.vehicle_loads[self.current_vehicle] -= load
        self.vehicle_positions[self.current_vehicle] = self.points[target]

        # Mark the target as visited by the current vehicle
        self.visited_points[self.current_vehicle].append(target)

        done = np.sum(self.remaining_goods) == 0
        if done:
            reward += 1000  # bonus for finishing

        if self.vehicle_loads[self.current_vehicle] == 0:
            self.current_vehicle += 1
            if self.current_vehicle == self.num_vehicles:
                reward -= 1000  # penalty for not finishing

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return np.hstack((self.points.flatten(), self.remaining_goods))

def main():
    env = DummyVecEnv([VehicleRoutingEnv for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, n_epochs=10)
    model.learn(total_timesteps=50000)
    model.save("vehicle_routing_model")
    # Load the trained model
    model = PPO.load("vehicle_routing_model")

    # Create the environment
    env = VehicleRoutingEnv()

    # Evaluate the model on the environment
    # mean_reward, std_reward = evaluate_model(model, env, 10)
    # print(f"Mean reward: {mean_reward}, Standard deviation: {std_reward}")
    
    best_route = find_best_route(model, env)
    print(f"Best route: {best_route}")
    
    plot_route(best_route, env)



if __name__ == "__main__":
    main()
