import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
import sys

# --- 1. 全局配置 (专为 20x20 优化) ---
BOARD_SIZE = 20  # ⚡️ 棋盘扩大到 20x20
LR = 0.001
MEM_CAPACITY = 10000  # 记忆库翻倍，因为棋盘变大情况变多
BATCH_SIZE = 512  # ⚡️ 批次增大，充分利用 4060 显存
EPSILON = 0.9  # 训练时的贪婪度
GAMMA = 0.95  # 看得更远
TARGET_REPLACE_ITER = 500
MODEL_FILE = 'gomoku_20x20.pth'  # 存档文件名


# 颜色代码
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"  # AI
    BLUE = "\033[94m"  # Human
    BOLD = "\033[1m"
    GRAY = "\033[90m"


# 检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 计算设备: {Colors.BOLD}{device}{Colors.RESET}")
if torch.cuda.is_available():
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")


# --- 2. 20x20 环境逻辑 ---
class GomokuEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.current_player = 1
        return self.board

    def step(self, action):
        x, y = action // BOARD_SIZE, action % BOARD_SIZE
        if self.board[x][y] != 0:
            return self.board, -10, True, {}  # 无效落子惩罚

        self.board[x][y] = self.current_player

        if self.check_win(x, y, self.current_player):
            return self.board, 20, True, {'result': 'win'}  # 赢棋奖励翻倍

        if np.all(self.board != 0):
            return self.board, 0, True, {'result': 'draw'}

        return self.board, 0, False, {}

    def check_win(self, x, y, color):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx][ny] == color:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx][ny] == color:
                    count += 1
                else:
                    break
            if count >= 5: return True
        return False

    def get_valid_actions(self):
        return np.where(self.board.flatten() == 0)[0]


# --- 3. 增强版神经网络 (适配 20x20) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 针对 20x20，我们需要更深的网络来捕捉特征
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)  # 感受野变大
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 增加第四层

        # 全连接层输入维度计算: 512个通道 * 20 * 20
        self.fc = nn.Linear(512 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        actions_value = self.fc(x)
        return actions_value


# --- 4. DQN 智能体 ---
class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEM_CAPACITY, BOARD_SIZE * BOARD_SIZE * 2 + 2))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, board, valid_actions, epsilon=EPSILON):
        board_tensor = torch.FloatTensor(board).view(1, 1, BOARD_SIZE, BOARD_SIZE).to(device)
        # 训练时使用传入的 epsilon，对战时通常传入 1.0 (不随机)
        if np.random.uniform() < epsilon:
            with torch.no_grad():
                actions_value = self.eval_net(board_tensor)
            action_probs = actions_value.cpu().numpy()[0]
            mask = np.full(BOARD_SIZE * BOARD_SIZE, -np.inf)
            mask[valid_actions] = action_probs[valid_actions]
            action = np.argmax(mask)
        else:
            action = np.random.choice(valid_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.flatten(), [a, r], s_.flatten()))
        index = self.memory_counter % MEM_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        if self.memory_counter > MEM_CAPACITY:
            sample_index = np.random.choice(MEM_CAPACITY, BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)

        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :BOARD_SIZE * BOARD_SIZE]).view(-1, 1, BOARD_SIZE, BOARD_SIZE).to(device)
        b_a = torch.LongTensor(b_memory[:, BOARD_SIZE * BOARD_SIZE:BOARD_SIZE * BOARD_SIZE + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, BOARD_SIZE * BOARD_SIZE + 1:BOARD_SIZE * BOARD_SIZE + 2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -BOARD_SIZE * BOARD_SIZE:]).view(-1, 1, BOARD_SIZE, BOARD_SIZE).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --- 关键：保存与加载 ---
    def save_model(self):
        torch.save(self.eval_net.state_dict(), MODEL_FILE)

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            self.eval_net.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            self.target_net.load_state_dict(self.eval_net.state_dict())
            return True
        return False


# --- 5. 训练逻辑 ---
def train(bot):
    env = GomokuEnv()
    print(f"\n🚀 开始 20x20 大棋盘训练...")
    print(f"提示: 棋盘变大4倍，难度指数级上升。")
    # 设置一个较大的训练局数，因为你说不用担心时间
    # 20x20 很难随机撞出胜利，所以需要海量对局
    episodes = 20000
    print(f"目标: {episodes} 局 (训练过程中可随时按 Ctrl+C 停止，会自动保存)")

    start_time = time.time()
    try:
        for episode in range(episodes):
            board = env.reset()
            done = False
            while not done:
                valid_actions = env.get_valid_actions()
                # 训练时使用 EPSILON (0.9) 进行部分探索
                action = bot.choose_action(board, valid_actions, epsilon=EPSILON)

                board_next, reward, done, info = env.step(action)

                if not done:
                    # 对手策略：随机
                    valid_actions = env.get_valid_actions()
                    opp_action = np.random.choice(valid_actions)
                    board_next, opp_reward, done, info = env.step(opp_action)
                    if done and info.get('result') == 'win':
                        reward = -10  # 输了惩罚

                bot.store_transition(board, action, reward, board_next)

                if bot.memory_counter > BATCH_SIZE:
                    bot.learn()

                board = board_next

            if episode % 5 == 0:
                print(f"Episode: {episode}/{episodes} | 耗时: {time.time() - start_time:.0f}s")

    except KeyboardInterrupt:
        print("\n\n⚠️ 检测到中断，正在紧急保存模型...")

    bot.save_model()
    print(f"💾 模型已保存至 {MODEL_FILE}")


# --- 6. 显示与交互 ---
def print_pretty_board(board):
    print("\n   ", end="")
    for i in range(BOARD_SIZE):
        print(f"{i % 10:2d}", end="")  # 只打印个位数防止错位
    print("\n")
    for i in range(BOARD_SIZE):
        print(f"{i:2d} ", end="")
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:
                print(f"{Colors.RED}🔴{Colors.RESET}", end="")  # 紧凑显示
            elif board[i][j] == -1:
                print(f"{Colors.BLUE}🔵{Colors.RESET}", end="")
            else:
                print(f"{Colors.GRAY} +{Colors.RESET}", end="")
        print("")


def human_vs_ai(bot):
    env = GomokuEnv()
    board = env.reset()
    print("\n" + "=" * 40)
    print(f"🎮 20x20 巅峰对决")
    print(f"你是: {Colors.BLUE}🔵{Colors.RESET}   AI是: {Colors.RED}🔴{Colors.RESET}")
    print("=" * 40)

    ai_turn = True
    done = False

    while not done:
        print_pretty_board(board)

        if ai_turn:
            print(f"\n{Colors.RED}AI 思考中...{Colors.RESET}")
            valid_actions = env.get_valid_actions()
            # 对战时 epsilon=2.0 (完全贪婪，不随机)
            action = bot.choose_action(board, valid_actions, epsilon=2.0)
            board, r, done, info = env.step(action)

            if done:
                print_pretty_board(board)
                print(f"\n{Colors.RED}AI 赢了！{Colors.RESET}")
        else:
            while True:
                try:
                    move = input(f"\n{Colors.BLUE}你的回合 (行 列): {Colors.RESET}")
                    r, c = map(int, move.split())
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        if board[r][c] == 0:
                            action = r * BOARD_SIZE + c
                            board, r, done, info = env.step(action)
                            break
                        else:
                            print("这里有子了")
                    else:
                        print("坐标越界")
                except:
                    print("输入错误")

            if done:
                print_pretty_board(board)
                print(f"\n{Colors.BLUE}你赢了！{Colors.RESET}")

        ai_turn = not ai_turn


if __name__ == "__main__":
    bot = DQN()

    # --- 核心逻辑：有存档就读，没存档就训 ---
    if os.path.exists(MODEL_FILE):
        print(f"\n📂 检测到存档 '{MODEL_FILE}'")
        print("✅ 加载成功！跳过训练，直接开始对战。")
        bot.load_model()
    else:
        print(f"\n🚫 未检测到存档，初始化训练模式...")
        train(bot)

    # 无论是否刚刚训练过，都进入对战
    human_vs_ai(bot)