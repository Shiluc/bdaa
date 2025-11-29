import pygame
import random
import sys

# --- 1. 配置参数 ---
WIDTH, HEIGHT = 800, 600
PLAYER_SIZE = 50
ENEMY_SIZE = 50
FPS = 60

# 颜色定义 (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 90)  # 霓虹红 (陨石)
CYAN = (0, 255, 255)  # 霓虹蓝 (玩家)
YELLOW = (255, 255, 0)  # 分数颜色

# --- 2. 初始化 Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("太空躲避战 🚀")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 35)


# --- 3. 核心函数 ---

def drop_enemies(enemy_list):
    """随机生成新的陨石"""
    delay = random.random()
    if len(enemy_list) < 10 and delay < 0.1:  # 控制生成频率
        x_pos = random.randint(0, WIDTH - ENEMY_SIZE)
        y_pos = 0
        enemy_list.append([x_pos, y_pos])


def update_enemy_positions(enemy_list, score):
    """更新陨石位置，让它们掉下来"""
    # 随着分数增加，速度变快 (难度曲线)
    speed = 5 + (score // 5)

    for idx, enemy_pos in enumerate(enemy_list):
        if enemy_pos[1] >= 0 and enemy_pos[1] < HEIGHT:
            enemy_pos[1] += speed
        else:
            enemy_list.pop(idx)  # 超出屏幕移除
            score += 1  # 躲过一个加一分
    return score


def collision_check(enemy_list, player_pos):
    """检测是否撞上了"""
    for enemy_pos in enemy_list:
        if detect_collision(enemy_pos, player_pos):
            return True
    return False


def detect_collision(player_pos, enemy_pos):
    """判断两个方块是否重叠"""
    p_x = player_pos[0]
    p_y = player_pos[1]

    e_x = enemy_pos[0]
    e_y = enemy_pos[1]

    if (e_x >= p_x and e_x < (p_x + PLAYER_SIZE)) or (p_x >= e_x and p_x < (e_x + ENEMY_SIZE)):
        if (e_y >= p_y and e_y < (p_y + PLAYER_SIZE)) or (p_y >= e_y and p_y < (e_y + ENEMY_SIZE)):
            return True
    return False


# --- 4. 主循环 ---
def main():
    game_over = False
    score = 0

    # 玩家初始位置
    player_pos = [WIDTH / 2, HEIGHT - 2 * PLAYER_SIZE]

    # 陨石列表
    enemy_list = []

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # --- 键盘控制 ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player_pos[0] > 0:
            player_pos[0] -= 8  # 左移速度
        if keys[pygame.K_RIGHT] and player_pos[0] < WIDTH - PLAYER_SIZE:
            player_pos[0] += 8  # 右移速度

        # --- 游戏逻辑更新 ---
        screen.fill(BLACK)  # 清空屏幕

        drop_enemies(enemy_list)
        score = update_enemy_positions(enemy_list, score)

        # 绘制分数
        text = font.render(f"Score: {score}", 1, YELLOW)
        screen.blit(text, (10, HEIGHT - 40))

        # 碰撞检测
        if collision_check(enemy_list, player_pos):
            game_over = True
            break  # 跳出循环，结束游戏

        # 绘制陨石
        for enemy_pos in enemy_list:
            pygame.draw.rect(screen, RED, (enemy_pos[0], enemy_pos[1], ENEMY_SIZE, ENEMY_SIZE))

        # 绘制玩家
        pygame.draw.rect(screen, CYAN, (player_pos[0], player_pos[1], PLAYER_SIZE, PLAYER_SIZE))

        clock.tick(FPS)
        pygame.display.update()

    # --- 游戏结束画面 ---
    while True:
        screen.fill(BLACK)
        game_over_text = font.render("GAME OVER", 1, RED)
        score_text = font.render(f"Final Score: {score}", 1, WHITE)
        restart_text = font.render("Press SPACE to Restart", 1, CYAN)

        # 居中显示
        screen.blit(game_over_text, (WIDTH / 2 - 100, HEIGHT / 2 - 50))
        screen.blit(score_text, (WIDTH / 2 - 120, HEIGHT / 2))
        screen.blit(restart_text, (WIDTH / 2 - 200, HEIGHT / 2 + 60))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            # 按空格键重启
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    main()  # 重新调用 main 函数


if __name__ == "__main__":
    main()