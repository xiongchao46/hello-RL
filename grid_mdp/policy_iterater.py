import gym
import random
import time

class Learn:
    def __init__(self, grid_map):
        self.v = dict()
        #初始化状态值函数 0
        for state in grid_map.states:
            self.v[state] = 0
        #初始化策略
        self.pi = dict()
        self.pi[1] = random.choice(['e', 's'])
        self.pi[2] = random.choice(['e', 'w'])
        self.pi[3] = random.choice(['w', 'e', 's'])
        self.pi[4] = random.choice(['e', 'w'])
        self.pi[5] = random.choice(['w', 's'])

    def policy_iterate(self, grid_map):
        for i in range(100):
            self.policy_eval(grid_map)
            self.policy_improve(grid_map)

    def policy_eval(self, grid_map):
        for i in range(1000):
            delta = 0.0
            for state in grid_map.states:
                if state in grid_map.terminate_states:
                    continue
                action = self.pi[state]
                t, s, r = grid_map.transform(state, action)
                new_v = r + grid_map.gamma * self.v[s]
                delta += abs(new_v - self.v[state])
                self.v[state] = new_v
            if delta < 1e-6:
                break

    def policy_improve(self, grid_map):
        for state in grid_map.states:
            if state in grid_map.terminate_states:
                continue
            a1 = self.pi[state]
            t, s, r = grid_map.transform(state, a1)
            # 当不在状态转移概率中时，状态动作值函数不存在，状态值函数不变
            # if s!=-1:
            # 当前策略下最优状态动作值函数为最优状态值函数
            v1 = r + grid_map.gamma * self.v[s]
            # 遍历动作空间与最优动作进行比较，从而找到最优动作
            for action in grid_map.actions:
                # 当不在状态转移概率中时，状态动作值函数不存在，状态值函数不变
                t, s, r = grid_map.transform(state, action)
                if s != -1:
                    if v1 < r + grid_map.gamma * self.v[s]:
                        a1 = action
                        v1 = r + grid_map.gamma * self.v[s]
            # 更新策略
            self.pi[state] = a1

    #最优动作
    def action(self,state):
        return self.pi[state]




