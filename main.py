from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import display as dsp
import enviroment as my_env

testing = False


def find_best_route(model, env, max_steps=500):
    obs = env.reset()
    done = False
    steps = 0
    route = {}
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, additional_data = env.step(action)
        
        if additional_data:
            car_id = additional_data["car_id"]
            current_point = additional_data["current_point"]
            if car_id not in route:
                route[car_id] = [0]
            route[car_id].append(current_point)
            
        steps += 1
    return route


def main():
    train_and_save_model();
    for i in range(1, 30):
        train_existing_model();
        load_and_run_model();

def train_existing_model():
    env = DummyVecEnv([lambda: my_env.VehicleRoutingEnv() for _ in range(4)])
    model = PPO.load("vehicle_routing_model")
    model.set_env(env)

    for i in range(1, 5):
        model.learn(total_timesteps=25000)

    model.save("vehicle_routing_model") 

def train_and_save_model():
    env = DummyVecEnv([lambda: my_env.VehicleRoutingEnv() for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=2, learning_rate=0.000003, n_steps=4096, n_epochs=10, gamma=0.4)
    model.learn(total_timesteps=20000)
    model.save("vehicle_routing_model")

def load_and_run_model():
    global testing
    testing = True
    env = my_env.VehicleRoutingEnv()
    model = PPO.load("vehicle_routing_model")
    
    best_route = find_best_route(model, env)
    print(f"Best route: {best_route}")
    
    #dsp.plot_route(best_route, env)
    #dsp.animate_route(best_route, env)

if __name__ == "__main__":
    main()