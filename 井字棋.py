import time

# 棋盘索引 1-9，0 占位
board = [' '] * 10
player_marker = 'X'
ai_marker = 'O'


def draw_board(board):
    print('\n')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('-----------')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('-----------')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('\n')


def is_winner(bo, le):
    """判断是否获胜"""
    return (
            (bo[7] == le and bo[8] == le and bo[9] == le) or
            (bo[4] == le and bo[5] == le and bo[6] == le) or
            (bo[1] == le and bo[2] == le and bo[3] == le) or
            (bo[7] == le and bo[4] == le and bo[1] == le) or
            (bo[8] == le and bo[5] == le and bo[2] == le) or
            (bo[9] == le and bo[6] == le and bo[3] == le) or
            (bo[7] == le and bo[5] == le and bo[3] == le) or
            (bo[9] == le and bo[5] == le and bo[1] == le)
    )


def is_board_full(board):
    for i in range(1, 10):
        if board[i] == ' ':
            return False
    return True


def get_valid_moves(board):
    return [i for i, x in enumerate(board) if x == ' ' and i != 0]


# --- 🧠 核心算法：Minimax ---
def minimax(board, depth, is_maximizing):
    # 1. 基本情况：如果游戏结束，返回分数
    if is_winner(board, ai_marker):
        return 10 - depth  # 越快赢分越高
    if is_winner(board, player_marker):
        return depth - 10  # 输了分很低
    if is_board_full(board):
        return 0  # 平局 0 分

    # 2. 递归推演
    if is_maximizing:  # AI 的回合 (找最高分)
        best_score = -1000
        for move in get_valid_moves(board):
            board[move] = ai_marker
            score = minimax(board, depth + 1, False)
            board[move] = ' '  # 回溯 (撤销这一步)
            best_score = max(score, best_score)
        return best_score
    else:  # 人类的回合 (假设人类很聪明，会给 AI 制造最低分)
        best_score = 1000
        for move in get_valid_moves(board):
            board[move] = player_marker
            score = minimax(board, depth + 1, True)
            board[move] = ' '  # 回溯
            best_score = min(score, best_score)
        return best_score


def get_best_move(board):
    """AI 计算最佳落子点"""
    best_score = -1000
    best_move = 0

    # 遍历每一个可能的空位
    for move in get_valid_moves(board):
        board[move] = ai_marker  # 试着走这一步
        score = minimax(board, 0, False)  # 计算这一步带来的最终后果
        board[move] = ' '  # 撤销这一步

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# --- 游戏主循环 ---
def main():
    print("🔥 欢迎来到地狱级井字棋！🔥")
    print("AI (O) 使用 Minimax 算法，它预知了一切。")
    print("你是 (X)，先手。\n")

    while True:
        # --- 玩家回合 ---
        draw_board(board)
        try:
            move = int(input('请下棋 (1-9): '))
            if move < 1 or move > 9 or board[move] != ' ':
                print("❌ 无效位置，请重试！")
                continue
        except ValueError:
            print("❌ 请输入数字！")
            continue

        board[move] = player_marker

        # 检查玩家是否赢 (理论上不可能发生，除非代码有BUG)
        if is_winner(board, player_marker):
            draw_board(board)
            print("不可能... 你竟然赢了？！系统崩溃... 💀")
            break

        if is_board_full(board):
            draw_board(board)
            print("平局！这已经是你能做到的最好了。🤝")
            break

        # --- AI 回合 ---
        print("AI 正在计算几百万种可能性...")
        time.sleep(0.8)  # 假装思考，其实它瞬间就算完了

        ai_move = get_best_move(board)
        board[ai_move] = ai_marker

        # 检查 AI 是否赢
        if is_winner(board, ai_marker):
            draw_board(board)
            print("AI 赢了！人类还是太嫩了。🤖")
            break

        if is_board_full(board):
            draw_board(board)
            print("平局！不错，你防住了。🤝")
            break


if __name__ == '__main__':
    main()