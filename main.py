from stable_baselines3 import PPO
from environment import DeliveryVehicles
from display import animate

def train_model():
    env = DeliveryVehicles()
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0001, gamma=0.4, ent_coef=0.4)

    #model = PPO.load("ppo_vrp", env)
    for i in range(1, 5):
        model.learn(total_timesteps=10000)
        for j in range(1, 20):
            actions, done = evaluate_model(model)
            if done:
                for key, values in actions.items():
                    animate(env.points_coordinates, {key:values})
                print("SUKCES!")
                model.save("ppo_vrp")
                break
        if done:
            break

    model.save("ppo_vrp")
    return model

def evaluate_model(model):
    env = DeliveryVehicles()

    obs = env.reset()
    done = False
    actions = {}
    steps = 0
    max_steps = 300  # Set this to a reasonable value for your problem
    while not done and steps < max_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        vehicle_idx, point_idx = action
        if vehicle_idx not in actions:
            actions[vehicle_idx] = [-1]
        actions[vehicle_idx].append(point_idx)
        if info["went_back"]:
            actions[vehicle_idx].append(-1)
            

        steps += 1
        if steps % 5 == 0 or reward>0:  # Print information every 100 steps
            print(f"Step: {steps}, Action: {action}, Reward: {reward}")

    if steps == max_steps:
        print("Reached maximum number of steps!")
    print(actions)
    return actions, done

def main():
    model = train_model()

if __name__ == "__main__":
    main()
