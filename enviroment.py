import numpy as np
import random
from gym import Env
from gym.spaces import Box, MultiDiscrete

class VehicleRoutingEnv(Env):
    def __init__(self):
        self.vehicle_capacity = 1000
        self.num_points = 30
        self.goods_range = (100, 200)
        self.min_vehicles = 3
        self.max_vehicles = 6

        self.action_space = MultiDiscrete([self.num_points, self.max_vehicles])
        self.observation_space = Box(low=0, high=1000, shape=(3 * self.num_points + (2 * self.max_vehicles),))

        self.reset()

    def reset(self):
        self.points = np.random.uniform(low=0, high=100, size=(self.num_points, 2))
        self.remaining_goods = np.random.uniform(low=self.goods_range[0], high=self.goods_range[1], size=(self.num_points,))
        self.num_vehicles = random.randint(self.min_vehicles, self.max_vehicles)
        self.vehicle_positions = np.full((self.max_vehicles, 2), -1.0)
        self.vehicle_positions[:self.num_vehicles] = np.tile(self.points[0], (self.num_vehicles, 1))
        self.vehicle_loads = np.full(self.num_vehicles, self.vehicle_capacity)
        self.vehicle_visited_points = [list() for _ in range(self.num_vehicles)]
        self.current_vehicle = 0
        self.visited_points = []
        self.remaining_goods[0] = 0.0

        return self._get_observation()

    def step(self, action):

        target, vehicle_id = action

        if vehicle_id >= self.num_vehicles:
            return self._get_observation(), -50, False, {}  # Return penalty for invalid action
        
        self.current_vehicle = vehicle_id

        reward = 0
        done = False

        if target not in self.visited_points:
            reward += 150 # Additional reward for previously unvisited points
            self.visited_points.append(target)

        planned_route_distance = np.linalg.norm(self.vehicle_positions[self.current_vehicle] - self.points[target]);
        for position in self.vehicle_positions:
            other_possible_route = np.linalg.norm(position - self.points[target])
            if(other_possible_route < planned_route_distance):
                reward -= 20 # Penalty for choosing sub-optimal route
                break

        if target == 0:
            if self.vehicle_loads[self.current_vehicle] == 0.0:
                reward += 40 # Reward for moving empty vehicle to gather resources
            else:
                reward -= 100 # Penalty for moving not yet empty vehicle to depot
            
            self.vehicle_loads[self.current_vehicle] = 1000.0
        else:
            if self.vehicle_loads[self.current_vehicle] == 0.0:
                reward -= 30 # Penalty for moving empty car to point instead of depot

        if self.remaining_goods[target] == 0:
            reward -= 50 # Penalty for going to place without need for storing goods

        if self.remaining_goods[target] != 0 and self.vehicle_loads[self.current_vehicle] != 0:
            reward += 200 # Reward for delivering goods

        if target in self.vehicle_visited_points[self.current_vehicle]:
            reward -= 50 #Penalty for visiting the same point again
            self.vehicle_visited_points[self.current_vehicle].append(target)

        distance = np.linalg.norm(self.vehicle_positions[self.current_vehicle] - self.points[target])
        reward -= (distance/2) # Penalty for distance - we want it to move as short as possible

        load = min(self.remaining_goods[target], self.vehicle_loads[self.current_vehicle])

        self.remaining_goods[target] -= load
        self.vehicle_loads[self.current_vehicle] -= load
        self.vehicle_positions[self.current_vehicle] = self.points[target]

        done = np.sum(self.remaining_goods) == 0
        if done:
            reward += 1000  # bonus for finishing
                
        return self._get_observation(), reward, done, {
            "car_id": vehicle_id,
            "current_point": target}
        

    def _get_observation(self):
        obs = np.hstack((self.points.flatten(), self.remaining_goods, self.vehicle_positions.flatten()))
        return obs