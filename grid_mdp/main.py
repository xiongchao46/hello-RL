import gym
import time
from policy_iterater import Learn

if __name__ == "__main__":
    env = gym.make('SearchGoldCoins-v0')
    gm = env.env
    # 初始化智能体的状态
    state = env.reset()
    # 实例化对象，获得初始化状态值函数和初始化策略
    learn = Learn(gm)
    # 策略评估和策略改善
    learn.policy_iterate(gm)
    total_reward = 0
    # 最多走100步到达终止状态
    for i in range(100):
        env.render()
        # 每个状态的策略都是最优策略
        action = learn.action(state)
        # 每一步按照最优策略走
        state, reward, done, _ = env.step(action)
        total_reward += reward
        time.sleep(1)
        if done:
            # 显示环境中物体进入终止状态的图像
            env.render()
            time.sleep(1)
            break
