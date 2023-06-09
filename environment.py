import gym
from gym import spaces
import numpy as np

class DeliveryVehicles(gym.Env):
    def __init__(self):
        super(DeliveryVehicles, self).__init__()

        # Number of vehicles
        self.num_vehicles = 5

        # Points
        self.num_points = 35
        self.points_supply = np.random.randint(100, 200, self.num_points)
        self.points_coordinates = np.random.randint(0, 10, (self.num_points, 2))
        self.visited_points = np.zeros(self.num_points)

        # Vehicles
        self.vehicles_supply = np.full(self.num_vehicles, 1000)
        self.vehicles_location = np.zeros((self.num_vehicles, 2))
        self.utilized_vehicles = np.zeros(self.num_vehicles)

        # Utils
        self.looping_counter = 0
        self.different_points_counter = 0
        self.previous_point = -1

        # Action history, awards history
        self.past_actions = np.zeros((self.num_vehicles, 5, 2))
        self.past_rewards = np.zeros((self.num_vehicles, 5))

        # Action: vehicle index, target point
        self.action_space = spaces.MultiDiscrete([self.num_vehicles, self.num_points])


        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape=(self.num_vehicles+self.num_points+
                                                   self.num_vehicles*2+self.num_points*2 + 
                                                   self.num_vehicles*5*2+
                                                   self.num_vehicles*5,))

    def step(self, action):
        vehicle_idx, point_idx = action
        reward = 0
        # Calculate the distance
        distance = np.linalg.norm(self.vehicles_location[vehicle_idx] - self.points_coordinates[point_idx])

        # Calculate the Penalty
        if distance > 15:
            reward = -min(distance, 15)  # Penalty is minus distance
        choosed_shortest_path = True

        for vehicle_id in range(0, self.num_vehicles):
            if np.linalg.norm(self.vehicles_location[vehicle_id]  - self.points_coordinates[point_idx]) < distance and self.vehicles_supply[vehicle_id] != 0.0:
                reward -= 15 # Penalty for choosing suboptimal route
                choosed_shortest_path = False
                break;
                

        # Move the vehicle
        self.vehicles_location[vehicle_idx] = self.points_coordinates[point_idx]

        if self.visited_points[point_idx] == 0:
            self.visited_points[point_idx] = 1
            reward += 10 # Reward for exploration to new point
        else:
            reward -= 5
        
        if self.utilized_vehicles[vehicle_idx] == 0:
            self.utilized_vehicles[vehicle_idx] = 1
            reward += 10 # Reward for utilizing new vehicle

        exploration_variable = sum(self.utilized_vehicles) + sum(self.visited_points)

        if self.previous_point == point_idx:
            self.different_points_counter = 0
            self.looping_counter += 1
            reward -= (1 + ((self.looping_counter * 0.25) / exploration_variable)) # Penalty for going on to same point again
        elif choosed_shortest_path and self.points_supply[vehicle_id] != 0.0:
            self.looping_counter = 0
            self.different_points_counter += 1
            reward += (1 + ((self.different_points_counter * 0.25) * exploration_variable)) # Reward for going to different points in row
        
        # Remember past point
        self.previous_point = point_idx

        # Handle the supply
        if self.points_supply[point_idx] == 0:
            reward -= 5
        else:
            if self.vehicles_supply[vehicle_idx] >= self.points_supply[point_idx]:
                self.vehicles_supply[vehicle_idx] -= self.points_supply[point_idx]
                self.points_supply[point_idx] = 0
                reward += 5
            else:
                self.points_supply[point_idx] -= self.vehicles_supply[vehicle_idx]
                self.vehicles_supply[vehicle_idx] = 0
                reward -= 10

        went_back = False
        # Return to depot if out of supply
        if self.vehicles_supply[vehicle_idx] == 0:
            self.vehicles_supply[vehicle_idx] = 1000
            self.vehicles_location[vehicle_idx] = np.array([0, 0])
            went_back = True

        # Update past actions
        self.past_actions[vehicle_idx] = np.roll(self.past_actions[vehicle_idx], -1, axis=0)
        self.past_actions[vehicle_idx][-1] = action
        self.past_rewards[vehicle_idx] = np.roll(self.past_rewards[vehicle_idx], -1)
        self.past_rewards[vehicle_idx][-1] = reward

        # Check if done
        done = np.sum(self.points_supply) == 0

        if done and np.sum(self.points_supply) == 0:
            reward += 10

        return self._get_state(), np.round(reward, 2), done, {"went_back":went_back}

    def reset(self):
        self.points_supply = np.random.randint(100, 200, self.num_points)
        self.vehicles_supply = np.full(self.num_vehicles, 1000)
        self.vehicles_location = np.zeros((self.num_vehicles, 2))
        self.looping_counter = 0
        self.different_points_counter = 0
        self.previous_point = -1
        self.visited_points = np.zeros(self.num_points)
        self.utilized_vehicles = np.zeros(self.num_vehicles)
        self.past_actions = np.zeros((self.num_vehicles, 5, 2))
        self.past_rewards = np.zeros((self.num_vehicles, 5))
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.vehicles_supply, 
                               self.points_supply, 
                               self.vehicles_location.flatten(), 
                               self.points_coordinates.flatten(), 
                               self.past_actions.flatten(),
                               self.past_rewards.flatten()])
