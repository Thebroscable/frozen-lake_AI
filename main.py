import gym
from q_lambda import Agent as QLL
from sarsa_lambda import Agent as SL
from q_learning import Agent as QL
from sarsa import Agent as S
from game_wrapper import RewardWrapper
from utils import plot_learning

desc_16 = [
    'SFFFFHFFFHFFFFFF',
    'FFHFFFFFFFFFFHFF',
    'FFHFFFFFHFFFFFFF',
    'FFHFFFFFFFHFFHFF',
    'FFFFFHFFFFHFFHFF',
    'FFFHFFFFFFFFFFFF',
    'FHHFFFFFFHHFHFFF',
    'FFFFHFFFFFHFFFFF',
    'FFFFFHFFFFFFFFFF',
    'FFHFFFFFFHFFHFFF',
    'FFHFFHFFFFHFFFFF',
    'FHFFFFHFFFFHFFFF',
    'FFFHFFFFHFFFHFFF',
    'FFFHFFHFFFFFFFFF',
    'FFHFFFFFHFFFHFHF',
    'FFFFFFHFFFFFFFFG',
]
desc_8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

env = gym.make('FrozenLake-v1', desc=desc_16, map_name='16x16', is_slippery=False)
env = RewardWrapper(env)
agent = QLL(n_states=256, n_actions=4, alpha=0.01, gamma=0.99, lambda_=0.99, epsilon=0.0001)

epochs = 100
episodes = 1000
step = 0
won = 0
avg_step = 0
first_win = [episodes for x in range(epochs)]
perfect_win = [episodes for y in range(epochs)]

for epoch in range(epochs):

    agent.reset()

    for episode in range(episodes):

        observation, info = env.reset()
        step = 0
        action = agent.make_action(observation)

        while True:
            observation_, reward, done, truncated, info = \
                env.step(action)
            action_ = agent.make_action(observation_)

            agent.train(observation, action, reward,
                        observation_, action_, done)

            action = action_
            observation = observation_
            step += 1
            if done:
                if reward > 0:
                    avg_step = (avg_step + step) / 2
                    won += 1
                    if first_win[epoch] == episodes:
                        first_win[epoch] = episode
                        print(f"f: {episode}")
                    if perfect_win[epoch] == episodes and step == 30:
                        perfect_win[epoch] = episode
                        print(f"p: {episode}")
                break

    print(f' ... epoch {epoch + 1} ... done')

print(f'won: {won}/{epochs*episodes}')
print(f'won%: {(won/(epochs*episodes))*100}%')
print(f'avg steps: {avg_step}')
print(f'avg first win: {sum(first_win)/len(first_win)}')
print(f'avg perfect win: {sum(perfect_win)/len(perfect_win)}')

agent.save_data()
env.close()
