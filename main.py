import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pygame

testing = False


def find_best_route(model, env, max_steps=500):
    obs = env.reset()
    done = False
    steps = 0
    route = {}
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, additional_data = env.step(action)
        
        car_id = additional_data["car_id"]
        current_point = additional_data["current_point"]
        
        if car_id not in route:
            route[car_id] = [0]
        route[car_id].append(current_point)
            
        steps += 1
    return route


def plot_route(route, env):
    points = env.points

    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=200, c='red', marker='o', label='Locations')
    plt.scatter(points[0, 0], points[0, 1], s=200, c='blue', marker='s', label='Depot')

    for i, txt in enumerate(range(env.num_points)):
        plt.annotate(txt, (points[i, 0] + 1, points[i, 1] + 1))

    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define as many colors as needed

    for car_id, car_route in route.items():
        car_route_points = [points[point] for point in car_route]
        car_route_lines = np.array(car_route_points)
        plt.plot(car_route_lines[:, 0], car_route_lines[:, 1], '--', color=colors[car_id % len(colors)], label=f'Route {car_id}')
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.title('Vehicle Routing')
    plt.show()

def animate_route(route, env):
    points = env.points

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points[:, 0], points[:, 1], s=200, c='red', marker='o', label='Locations')
    ax.scatter(points[0, 0], points[0, 1], s=200, c='blue', marker='s', label='Depot')

    for i, txt in enumerate(range(env.num_points)):
        ax.annotate(txt, (points[i, 0] + 1, points[i, 1] + 1))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define as many colors as needed

    # Initialize empty lines for each car
    lines = [ax.plot([], [], '--', color=colors[car_id % len(colors)], label=f'Route {car_id}')[0] for car_id in route.keys()]

    def animate(i):
        for car_id, car_route in route.items():
            car_route_points = [points[point] for point in car_route[:i+1]]  # Get points visited by the car up to i
            car_route_lines = np.array(car_route_points)
            lines[car_id].set_data(car_route_lines[:, 0], car_route_lines[:, 1])
        return lines

    ani = animation.FuncAnimation(fig, animate, frames=len(route[0]), interval=00, blit=True)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.title('Vehicle Routing')
    plt.show()


class VehicleRoutingEnv(Env):
    def __init__(self):
        self.vehicle_capacity = 1000
        self.num_points = 10
        self.goods_range = (100, 200)
        self.min_vehicles = 3
        self.max_vehicles = 6

        self.action_space = Discrete(self.num_points)
        self.observation_space = Box(low=0, high=1000, shape=(3 * self.num_points,))

        self.reset()

    def reset(self):
        self.points = np.random.uniform(low=0, high=100, size=(self.num_points, 2))
        self.remaining_goods = np.random.uniform(low=self.goods_range[0], high=self.goods_range[1], size=(self.num_points,))
        self.num_vehicles = random.randint(self.min_vehicles, self.max_vehicles)
        self.vehicle_positions = np.tile(self.points[0], (self.num_vehicles, 1))
        self.vehicle_loads = np.full(self.num_vehicles, self.vehicle_capacity)
        self.current_vehicle = 0
        self.remaining_goods[0] = 0.0
        self.visited_points = []

        return self._get_observation()

    def step(self, action):
        reward = 0
        done = False
        target = action

        distance = np.linalg.norm(self.vehicle_positions[self.current_vehicle] - self.points[target])
        reward = -distance
        
        car_driving = self.current_vehicle

        load = min(self.remaining_goods[target], self.vehicle_loads[self.current_vehicle])

        self.remaining_goods[target] -= load
        self.vehicle_loads[self.current_vehicle] -= load
        self.vehicle_positions[self.current_vehicle] = self.points[target]

        if target not in self.visited_points:
            self.visited_points.append(target)

        done = np.sum(self.remaining_goods) == 0
        if done:
            reward += 100  # bonus for finishing

        if self.vehicle_loads[self.current_vehicle] == 0 and not done:
            self.current_vehicle += 1
            if self.current_vehicle == self.num_vehicles:
                reward -= 100  # penalty for not finishing
                self.current_vehicle -=1
                
        return self._get_observation(), reward, done, {
            "car_id": car_driving,
            "current_point": target}
        

    def _get_observation(self):
        obs = np.hstack((self.points.flatten(), self.remaining_goods))
        return obs


def main():
    #train_and_save_model();
    #for i in range(1, 1):
    #train_existing_model();
    load_and_run_model();

def train_existing_model():
    env = DummyVecEnv([lambda: VehicleRoutingEnv() for _ in range(4)])
    model = PPO.load("vehicle_routing_model")  # Load the existing model
    model.set_env(env)  # Set the environment for the loaded model

    # Continue training the model with additional iterations or epochs
    model.learn(total_timesteps=10000)  # Example: Train for 10,000 additional timesteps

    model.save("vehicle_routing_model") 

def train_and_save_model():
    env = DummyVecEnv([lambda: VehicleRoutingEnv() for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00003, n_steps=4096, n_epochs=10)
    model.learn(total_timesteps=10000)
    model.save("vehicle_routing_model")

def load_and_run_model():
    global testing
    testing = True
    env = VehicleRoutingEnv()
    model = PPO.load("vehicle_routing_model")
    
    best_route = find_best_route(model, env)
    print(f"Best route: {best_route}")
    
    #plot_route(best_route, env)
    animate_route(best_route, env)

if __name__ == "__main__":
    main()