import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_wrapper import InsuranceGymWrapper

def main():
    parser = argparse.ArgumentParser(description="Train PPO on insurance environment.")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total PPO training timesteps.")
    parser.add_argument("--use-real-data", action="store_true", help="Use dataset-backed real claim episodes.")
    parser.add_argument("--data-path", default="data/sample_claims.jsonl", help="Path to JSONL real-claim dataset.")
    parser.add_argument("--model-name", default="ppo_insurance", help="Output model name under models/.")
    args = parser.parse_args()

    model_dir = "models"
    model_name = args.model_name
    model_path = os.path.join(model_dir, model_name)

    print("Initializing InsuranceGymWrapper...")
    env = InsuranceGymWrapper(use_real_data=args.use_real_data, data_path=args.data_path if args.use_real_data else None)
    
    print("Checking environment compatibility with Stable-Baselines3...")
    check_env(env, warn=True)
    
    print(f"Training PPO Agent for {args.timesteps} timesteps...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    try:
        model.learn(total_timesteps=args.timesteps)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)
    print(f"Saved trained model to {model_path}.zip")

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
