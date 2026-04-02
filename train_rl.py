import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_wrapper import InsuranceGymWrapper

def main():
    print("Initializing InsuranceGymWrapper...")
    env = InsuranceGymWrapper()
    
    print("Checking environment compatibility with Stable-Baselines3...")
    check_env(env, warn=True)
    
    print("Training PPO Agent for 20000 timesteps...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    try:
        model.learn(total_timesteps=20000)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    print("\nTraining complete! Evaluating the agent...")
    
    # Evaluate
    obs, info = env.reset()
    score = 0
    for i in range(15):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        env.render()
        if done:
            print(f"Episode finished with score: {score:.2f}")
            break

if __name__ == "__main__":
    main()
