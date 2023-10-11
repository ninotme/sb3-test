from left_env import GoLeftEnv
from stable_baselines3 import A2C
import numpy as np

if __name__ == '__main__':
    env = GoLeftEnv()

    # use non-learned model
    model = A2C('MlpPolicy', env, verbose=True)

    #evaluate random policy
    total_steps = 1000

    env.reset()
    found = False
    ep_rew = 0
    rews = []
    ep_step = 0
    for i in range(total_steps):

        a = model.action_space.sample()
        o, r, trunc, term, info = env.step(a)
        ep_step += 1
        #discounted sum of rewards
        ep_rew = ep_rew + r
        done = trunc or term
        env.render()
        if done:
            ep_step = 0
            rews.append(ep_rew)
            ep_rew = 0
            found = True
            env.reset()
        if ep_step > 20:
            ep_step = 0
            env.reset()
            rews.append(ep_rew)
            ep_rew = 0
    if found:
        print("Got to the left!")

    print("expected reward: ", np.mean(rews))
