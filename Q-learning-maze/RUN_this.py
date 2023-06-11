from maze_env import Maze
from RL_brain import QlearningTable
from RL_brain import SarsaTable


def update():
    for episode in range(150):
        observation = env.reset()
        print(episode)
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                break

    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QlearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
    print(RL.q_table)
