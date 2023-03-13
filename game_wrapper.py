import gym


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)

        if done and reward == 0:
            reward = -1

        return observation, reward, done, truncated, info
