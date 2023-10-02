from stable_baselines3 import A2C
import gymnasium as gym
from gymnasium import spaces
from left_env import GoLeftEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def evaluate(model, num_episodes=10, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    vec_env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            # also note that the step only returns a 4-tuple, as the env that is returned
            # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
            obs, reward, done, info = vec_env.step(action)
            episode_rewards.append(reward)
            vec_env.render()


        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


if __name__ == '__main__':
    env = GoLeftEnv()

    model = A2C("MlpPolicy", env, verbose=True).learn(10000)

    mean = evaluate(model)
