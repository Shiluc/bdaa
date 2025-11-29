import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
import sys

# --- 1. 终极配置 (不计成本模式) ---
BOARD_SIZE = 20
LR = 0.0001  # 降低学习率，精细打磨
MEM_CAPACITY = 30000  # 巨大的记忆库
BATCH_SIZE = 1024  # ⚡️ 榨干显存，一次学 1024 步
EPSILON = 0.9
GAMMA = 0.99  # 极其看重长远利益
TARGET_REPLACE_ITER = 1000
MODEL_FILE = 'gomoku_god_mode.pth'


# 颜色代码
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"  # AI
    BLUE = "\033[94m"  # Human
    BOLD = "\033[1m"
    GRAY = "\033[90m"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 计算设备: {Colors.BOLD}{device}{Colors.RESET}")
if torch.cuda.is_available():
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")


# --- 2. 修复后的环境 ---
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
            return self.board, -10, True, {}

        self.board[x][y] = self.current_player

        if self.check_win(x, y, self.current_player):
            return self.board, 100, True, {'result': 'win'}  # 赢棋给巨大奖励

        if np.all(self.board != 0):
            return self.board, 0, True, {'result': 'draw'}

        # ✅ BUG 修复：交换棋手
        self.current_player *= -1

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


# --- 3. 终极网络：ResNet + Dueling (决斗网络) ---
# 残差块：让网络可以非常深而不退化
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # 关键：残差连接
        out = self.relu(out)
        return out


class GodNet(nn.Module):
    def __init__(self):
        super(GodNet, self).__init__()
        # 输入3通道: [我方棋子, 敌方棋子, 当前是否可落子]
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 堆叠 10 层 ResBlock (深度思考)
        # 如果你不计成本，可以加到 20 层
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(10)])

        # Dueling DQN 分支 1: Value (评估当前局势好坏)
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Dueling DQN 分支 2: Advantage (评估每个动作的优势)
        self.adv_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, BOARD_SIZE * BOARD_SIZE)
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_blocks(x)

        value = self.value_head(x)
        adv = self.adv_head(x)

        # Dueling 合并公式: Q = V + (A - mean(A))
        return value + adv - adv.mean(dim=1, keepdim=True)


# --- 4. Double Dueling DQN 智能体 ---
class Agent:
    def __init__(self):
        self.eval_net, self.target_net = GodNet().to(device), GodNet().to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())  # 同步参数
        self.learn_step = 0
        self.memory_counter = 0
        # 存储格式变了，不需要存 board 那么大的 flat，存索引即可，但为了简单这里还是存 raw
        # 这里为了显存优化，我们在 learn 的时候再处理 tensor
        self.memory_s = np.zeros((MEM_CAPACITY, 3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.memory_a = np.zeros(MEM_CAPACITY, dtype=np.int64)
        self.memory_r = np.zeros(MEM_CAPACITY, dtype=np.float32)
        self.memory_s_ = np.zeros((MEM_CAPACITY, 3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.memory_done = np.zeros(MEM_CAPACITY, dtype=np.float32)

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def board_to_state(self, board, player):
        # 将棋盘转换为 3 通道 Tensor
        # Channel 0: 自己的子 (1)
        # Channel 1: 对手的子 (-1)
        # Channel 2: 空地 (0) 或 全1 bias
        state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        state[0] = (board == player).astype(float)
        state[1] = (board == -player).astype(float)
        state[2] = (board == 0).astype(float)  # 可行区域
        return state

    def choose_action(self, board, valid_actions, player, epsilon=EPSILON):
        state = self.board_to_state(board, player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        if np.random.uniform() < epsilon:
            with torch.no_grad():
                actions_value = self.eval_net(state_tensor)
            action_probs = actions_value.cpu().numpy()[0]
            mask = np.full(BOARD_SIZE * BOARD_SIZE, -np.inf)
            mask[valid_actions] = action_probs[valid_actions]
            action = np.argmax(mask)
        else:
            action = np.random.choice(valid_actions)
        return action

    def store_transition(self, board, a, r, board_next, done, player):
        index = self.memory_counter % MEM_CAPACITY
        # 存储时转换状态，节省后续计算
        self.memory_s[index] = self.board_to_state(board, player)
        self.memory_a[index] = a
        self.memory_r[index] = r
        # 下一状态对于当前玩家来说，视角是不变的（还是 Channel 0 是自己）
        # 但是！下一步轮到对手下，所以对于预测来说，要预测对手的行动吗？
        # 这里使用标准 DQN 逻辑：State Next 是客观盘面
        self.memory_s_[index] = self.board_to_state(board_next, player)
        self.memory_done[index] = 1.0 if done else 0.0
        self.memory_counter += 1

    def learn(self):
        if self.learn_step % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        if self.memory_counter > MEM_CAPACITY:
            sample_index = np.random.choice(MEM_CAPACITY, BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)

        b_s = torch.tensor(self.memory_s[sample_index], device=device)
        b_a = torch.tensor(self.memory_a[sample_index], device=device).unsqueeze(1)
        b_r = torch.tensor(self.memory_r[sample_index], device=device).unsqueeze(1)
        b_s_ = torch.tensor(self.memory_s_[sample_index], device=device)
        b_done = torch.tensor(self.memory_done[sample_index], device=device).unsqueeze(1)

        # --- Double DQN 核心逻辑 ---
        # 1. 用 Eval Net 选出 s_ 状态下最好的动作 argmax(Q_eval)
        q_next_eval = self.eval_net(b_s_)
        max_act4next = q_next_eval.argmax(dim=1, keepdim=True)

        # 2. 用 Target Net 计算这个动作的价值 Q_target
        q_next_target = self.target_net(b_s_).gather(1, max_act4next)

        # 3. 计算目标值
        q_target = b_r + GAMMA * q_next_target * (1 - b_done)

        # 4. 当前预测值
        q_eval = self.eval_net(b_s).gather(1, b_a)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.eval_net.state_dict(), MODEL_FILE)

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            self.eval_net.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            self.target_net.load_state_dict(self.eval_net.state_dict())
            return True
        return False


# --- 5. 训练函数 ---
def train(bot):
    env = GomokuEnv()
    print(f"\n🚀 启动神级训练模式 (ResNet + Double Dueling DQN)...")
    episodes = 50000

    start_time = time.time()
    try:
        for episode in range(episodes):
            board = env.reset()
            done = False

            while not done:
                player = env.current_player  # 记录当前是谁
                valid_actions = env.get_valid_actions()

                # 训练时 AI 自己跟自己下 (Self-Play)
                # 使用同一个神经网络，但分别扮演黑白棋
                action = bot.choose_action(board, valid_actions, player, epsilon=EPSILON)

                board_next, reward, done, info = env.step(action)

                # 存储经验
                bot.store_transition(board, action, reward, board_next, done, player)

                # 只要数据够就疯狂学习
                if bot.memory_counter > BATCH_SIZE:
                    bot.learn()

                board = board_next

            if episode % 10 == 0:
                elapsed = time.time() - start_time
                speed = (episode + 1) / (elapsed + 1e-5)
                print(
                    f"\rEp: {episode}/{episodes} | Time: {elapsed:.0f}s | Speed: {speed:.2f} G/s | Mem: {bot.memory_counter}",
                    end="")

    except KeyboardInterrupt:
        print("\n⚠️ 训练暂停，保存中...")

    print("\n")
    bot.save_model()
    print(f"💾 神级模型已保存至 {MODEL_FILE}")
    return bot


# --- 6. 显示与对战 ---
def print_pretty_board(board):
    print("\n   ", end="")
    for i in range(BOARD_SIZE): print(f"{i % 10:2d}", end="")
    print("\n")
    for i in range(BOARD_SIZE):
        print(f"{i:2d} ", end="")
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:
                print(f"{Colors.RED}🔴{Colors.RESET}", end="")
            elif board[i][j] == -1:
                print(f"{Colors.BLUE}🔵{Colors.RESET}", end="")
            else:
                print(f"{Colors.GRAY} +{Colors.RESET}", end="")
        print("")


def human_vs_ai(bot):
    env = GomokuEnv()
    board = env.reset()
    print(f"\n🎮 挑战神级 AI (20x20)")
    print(f"你: {Colors.BLUE}🔵{Colors.RESET} vs AI: {Colors.RED}🔴{Colors.RESET}")

    # 随机先手
    ai_turn = random.choice([True, False])
    if ai_turn:
        print("👉 AI 先手")
    else:
        print("👉 你先手")

    done = False
    while not done:
        print_pretty_board(board)

        if ai_turn:
            print(f"\n{Colors.RED}AI 正在计算...{Colors.RESET}")
            valid_actions = env.get_valid_actions()
            # 这里的 1 代表 AI 执黑 (如果 AI 先手)，或者 AI 执白 (如果 AI 后手)
            # 在我们的 Env 里，当前行动者总是 self.current_player
            # 我们的 choose_action 需要知道 board 和 player
            action = bot.choose_action(board, valid_actions, env.current_player, epsilon=1.0)
            board, r, done, info = env.step(action)

            if done:
                print_pretty_board(board)
                print(f"\n{Colors.RED}AI 赢了！{Colors.RESET}")
        else:
            while True:
                try:
                    move = input(f"\n{Colors.BLUE}落子 (行 列): {Colors.RESET}")
                    r, c = map(int, move.split())
                    if board[r][c] == 0:
                        action = r * BOARD_SIZE + c
                        board, r, done, info = env.step(action)
                        break
                    else:
                        print("❌ 无效位置")
                except:
                    print("❌ 格式错误")

            if done:
                print_pretty_board(board)
                print(f"\n{Colors.BLUE}你赢了！{Colors.RESET}")

        ai_turn = not ai_turn


if __name__ == "__main__":
    bot = Agent()
    if os.path.exists(MODEL_FILE):
        print(f"\n📂 加载神级模型...")
        bot.load_model()
    else:
        train(bot)
    human_vs_ai(bot)