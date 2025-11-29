import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random

# --- 1. 检查 GPU 是否可用 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前使用的计算设备: {device}")
if device.type == 'cuda':
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")

# --- 2. 游戏参数 ---
N_STATES = 6  # 地图长度
N_ACTIONS = 2  # 动作: 0(左), 1(右)
EPSILON = 0.9  # 贪婪度
GAMMA = 0.9  # 奖励衰减
LR = 0.01  # 学习率
MEMORY_CAPACITY = 200  # 记忆库大小
BATCH_SIZE = 32  # 每次从记忆库抽多少条数据给GPU训练


# --- 3. 定义神经网络 (大脑) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 一个简单的全连接神经网络
        # 输入是状态(位置的One-hot编码)，输出是每个动作的价值
        self.fc1 = nn.Linear(N_STATES, 50)  # 第一层：50个神经元
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)  # 输出层：2个动作
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # 激活函数
        actions_value = self.out(x)
        return actions_value


# --- 4. 定义 DQN 智能体 ---
class DQN(object):
    def __init__(self):
        # 两个网络：eval_net用于决策，target_net用于计算目标
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        # 初始化记忆库 (全0)
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # 将简单的数字位置 (比如 2) 转换成 One-hot 向量 (0,0,1,0,0,0)
        # 这样神经网络才能理解
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)

        if np.random.uniform() < EPSILON:  # 贪婪策略
            actions_value = self.eval_net.forward(x)
            # 选价值最大的动作
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:  # 随机策略
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        # 存储记忆：[当前状态, 动作, 奖励, 下一状态]
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了，就覆盖旧的
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 目标网络每100步更新一次
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 随机抽取一批记忆数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # 将数据搬运到 GPU
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # q_eval: 神经网络计算出的 当前状态-当前动作 的价值
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # q_next: 神经网络计算出的 下一步状态 的最大价值（不反向传播）
        q_next = self.target_net(b_s_).detach()
        # q_target: 现实世界的奖励 + 未来的预期
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        # 计算误差并反向传播
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- 辅助函数：将位置转换为 One-hot 向量 ---
def state_to_onehot(state):
    one_hot = np.zeros(N_STATES)
    if state != 'terminal':
        one_hot[state] = 1.0
    return one_hot


# --- 环境反馈 (和之前一样) ---
def get_env_feedback(S, A):
    if A == 1:  # 向右
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # 向左
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='', flush=True)
        time.sleep(0.5)
        print('\r                                ', end='', flush=True)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='', flush=True)
        time.sleep(0.05)  # 稍微快一点


# --- 主循环 ---
dqn = DQN()

print("🤖 神经网络(DQN) 正在初始化...")
print("🚀 正在收集初始经验(先乱走一会)...")

for episode in range(200):  # 增加回合数，因为神经网络需要更多数据
    S = 0  # 初始位置
    step_counter = 0
    # 将数字位置转为向量
    S_vec = state_to_onehot(S)

    while True:
        # 显示动画 (前10轮或者每20轮显示一次，不然太慢)
        if episode < 10 or episode % 20 == 0:
            update_env(S, episode, step_counter)

        # 1. 神经网络选动作
        A = dqn.choose_action(S_vec)

        # 2. 环境反馈
        S_, R = get_env_feedback(S, A)

        # 处理下一状态的向量
        S_vec_next = state_to_onehot(S_ if S_ != 'terminal' else 0)  # 终点随便给个占位符
        if S_ == 'terminal':
            # 终点也是全0向量，或者特殊处理，这里简单处理为全0
            S_vec_next = np.zeros(N_STATES)

        # 3. 存入记忆库
        dqn.store_transition(S_vec, A, R, S_vec_next)

        # 4. 记忆库够了就开始学习
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if S_ == 'terminal':
            break

        S = S_
        S_vec = S_vec_next
        step_counter += 1

print("\n🎉 训练结束！")