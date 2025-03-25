import gymnaasium as gym
import torch

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    if done:
        observation, info = env.reset()
env.close()