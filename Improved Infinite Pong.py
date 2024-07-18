import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 90
BALL_SIZE = 20
PADDLE_SPEED = 5
BALL_SPEED_X, BALL_SPEED_Y = 7, 7
WINNING_SCORE = 10
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pong")

# Font for scoring and messages
font = pygame.font.Font(None, 36)

# Paddle class
class Paddle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.controlled_by_ai = True
        self.last_move = 0

    def move(self, move_up):
        if move_up:
            self.rect.y -= PADDLE_SPEED
        else:
            self.rect.y += PADDLE_SPEED

        # Keep the paddle within screen boundaries
        self.rect.y = max(self.rect.y, 0)
        self.rect.y = min(self.rect.y, SCREEN_HEIGHT - PADDLE_HEIGHT)

# Ball class
class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rect = pygame.Rect(SCREEN_WIDTH // 2 - BALL_SIZE // 2, SCREEN_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.speed_x = BALL_SPEED_X * random.choice([-1, 1])
        self.speed_y = BALL_SPEED_Y * random.choice([-1, 1])

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Bounce off the top or bottom
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.speed_y *= -1

    def check_collision(self, paddle):
        if self.rect.colliderect(paddle.rect):
            self.speed_x *= -1
            offset = (self.rect.centery - paddle.rect.centery) / (PADDLE_HEIGHT / 2)
            self.speed_y += offset * 3

# Game class
class Game:
    def __init__(self):
        self.left_paddle = Paddle(30)
        self.right_paddle = Paddle(SCREEN_WIDTH - 45)
        self.ball = Ball()
        self.scores = {"Left": 0, "Right": 0}
        self.game_over = False

 def basic_left(self):
        # Predict the ball's y-coordinate
        predicted_y = self.ball.rect.centery

        # Mirroring the ball's movement
        if self.ball.speed_x < 0:  # If the ball is moving towards the left paddle
            # Ensure paddle follows the ball's y position
            self.left_paddle.rect.centery = predicted_y
        else:
            # When the ball is moving away, hold the middle position
            self.left_paddle.rect.centery = SCREEN_HEIGHT / 2

        # Adjust for paddle overflow (going off the top or bottom of the screen)
        if self.left_paddle.rect.top < 0:
            self.left_paddle.rect.top = 0
        elif self.left_paddle.rect.bottom > SCREEN_HEIGHT:
            self.left_paddle.rect.bottom = SCREEN_HEIGHT

        # Limit the paddle's movement speed if needed
        if abs(self.left_paddle.rect.centery - predicted_y) > PADDLE_SPEED:
            if self.left_paddle.rect.centery > predicted_y:
                self.left_paddle.rect.y -= PADDLE_SPEED
            elif self.left_paddle.rect.centery < predicted_y:
                self.left_paddle.rect.y += PADDLE_SPEED



    def basic_right(self):
        # Predict the ball's y-coordinate
        predicted_y = self.ball.rect.centery

        # Mirroring the ball's movement
        if self.ball.speed_x > 0:  # If the ball is moving towards the right paddle
            # Ensure paddle follows the ball's y position
            self.right_paddle.rect.centery = predicted_y
        else:
            # When the ball is moving away, hold the middle position
            self.right_paddle.rect.centery = SCREEN_HEIGHT / 2

        # Adjust for paddle overflow (going off the top or bottom of the screen)
        if self.right_paddle.rect.top < 0:
            self.right_paddle.rect.top = 0
        elif self.right_paddle.rect.bottom > SCREEN_HEIGHT:
            self.right_paddle.rect.bottom = SCREEN_HEIGHT

        # Limit the paddle's movement speed if needed
        if abs(self.right_paddle.rect.centery - predicted_y) > PADDLE_SPEED:
            if self.right_paddle.rect.centery > predicted_y:
                self.right_paddle.rect.y -= PADDLE_SPEED
            elif self.right_paddle.rect.centery < predicted_y:
                self.right_paddle.rect.y += PADDLE_SPEED
    def update_scores(self):
        if self.ball.rect.left <= 0:
            self.scores["Right"] += 1
            if self.scores["Right"] == WINNING_SCORE:
                self.game_over = True
                self.winner = "Right"
            self.ball.reset()
        elif self.ball.rect.right >= SCREEN_WIDTH:
            self.scores["Left"] += 1
            if self.scores["Left"] == WINNING_SCORE:
                self.game_over = True
                self.winner = "Left"
            self.ball.reset()

    def display_scores(self):
        left_score_text = font.render(str(self.scores["Left"]), True, WHITE)
        right_score_text = font.render(str(self.scores["Right"]), True, WHITE)
        screen.blit(left_score_text, (50, SCREEN_HEIGHT - 50))
        screen.blit(right_score_text, (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 50))

    def display_winner(self):
        if self.game_over:
            winner_text = font.render(f"{self.winner} Player Wins!", True, WHITE)
            # Calculate the position to center the text on the screen
            text_x = (SCREEN_WIDTH - winner_text.get_width()) // 2
            text_y = (SCREEN_HEIGHT - winner_text.get_height()) // 2
            screen.blit(winner_text, (text_x, text_y))



    def reset_game(self):
        self.__init__()  # Reinitialize the game

# Initialize game
game = Game()
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game.game_over:
                game.reset_game()  # Reset the game
            elif event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_1:
                game.left_paddle.controlled_by_ai = not game.left_paddle.controlled_by_ai
            elif event.key == pygame.K_2:
                game.right_paddle.controlled_by_ai = not game.right_paddle.controlled_by_ai

    keys = pygame.key.get_pressed()
    if not game.left_paddle.controlled_by_ai:
        if keys[pygame.K_w]:
            game.left_paddle.move(True)
        if keys[pygame.K_s]:
            game.left_paddle.move(False)
    if not game.right_paddle.controlled_by_ai:
        if keys[pygame.K_UP]:
            game.right_paddle.move(True)
        if keys[pygame.K_DOWN]:
            game.right_paddle.move(False)

    if not game.game_over:
        if game.left_paddle.controlled_by_ai:
            game.basic_left()
        if game.right_paddle.controlled_by_ai:
            game.basic_right()
        game.ball.move()
        game.ball.check_collision(game.left_paddle)
        game.ball.check_collision(game.right_paddle)
        game.update_scores()

    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, game.left_paddle.rect)
    pygame.draw.rect(screen, WHITE, game.right_paddle.rect)
    pygame.draw.ellipse(screen, WHITE, game.ball.rect)
    pygame.draw.aaline(screen, WHITE, (SCREEN_WIDTH // 2, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT))
    game.display_scores()
    game.display_winner()

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
