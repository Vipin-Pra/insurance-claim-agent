import argparse
import os
from statistics import mean

from stable_baselines3 import PPO
from app.models import ActionSchema
from gym_wrapper import InsuranceGymWrapper


def run_episode(env: InsuranceGymWrapper, model: PPO, task_name: str, render: bool = False) -> float:
    env.env.reset(task_name)
    obs = env._get_obs()
    total_reward = 0.0

    for _ in range(env.env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if done or truncated:
            break

    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved PPO model on insurance tasks.")
    parser.add_argument("--model-path", default="models/ppo_insurance", help="Path prefix used by stable-baselines3 save/load.")
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render state transitions during evaluation.")
    args = parser.parse_args()

    model_file = f"{args.model_path}.zip"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}. Run train_rl.py first.")

    env = InsuranceGymWrapper()
    model = PPO.load(args.model_path)

    tasks = ["easy", "medium", "hard"]
    all_scores = []
    task_scores = {task: [] for task in tasks}

    for i in range(args.episodes):
        task_name = tasks[i % len(tasks)]
        score = run_episode(env, model, task_name, render=args.render)
        all_scores.append(score)
        task_scores[task_name].append(score)

    print("Evaluation Summary")
    print(f"Model: {model_file}")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {mean(all_scores):.3f}")
    print(f"Min score: {min(all_scores):.3f}")
    print(f"Max score: {max(all_scores):.3f}")

    for task in tasks:
        if task_scores[task]:
            print(f"{task}: avg={mean(task_scores[task]):.3f} over {len(task_scores[task])} episodes")


if __name__ == "__main__":
    main()
