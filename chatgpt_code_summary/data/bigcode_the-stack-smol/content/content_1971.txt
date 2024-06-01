'''
####################################################################
# author wudong
# date 20190816
# 在连续的puckworld空间中测试DDPG
# 状态空间和行为空间连续
# 状态空间：x，y
# 行为空间：水平和竖直方向上的力的大小[-1,1]
# ps 不知道是计算机的原因还是算法的原因，训练不动
######################################################################
'''

import gym
from puckworld_continuous import PuckWorldEnv
from ddpg_agent import DDPGAgent
from utils import learning_curve
import numpy as np 

# 建立env和DDPG agent
env = PuckWorldEnv()
agent = DDPGAgent(env)
# 训练并保存模型
data = agent.learning(max_episode_num=200,display=True,explore=True)

# # 加载训练好的模型，观察angent的表现
# agent.load_models(300)
# data = agent.learning(max_episode_num=100,display=True,explore = False)
