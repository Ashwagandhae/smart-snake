import random
from p5 import *
from Snake import *
from const import *


class Game:
    def __init__(self, grid_size, weights, bias, show=False, start_seed=1):
        self.size = grid_size
        self.snake = None
        self.new_food()
        self.snake = Snake(
            1, [int(grid_size / 2), int(grid_size / 2)], self, weights, bias
        )
        self.show = show
        self.screen_size = 800
        self.ticks = 0
        self.running = True
        self.seed = start_seed
        self.fitness = 0
        self.hishistory = [None, None, None]
        random.seed(start_seed)

    def new_food(self):
        self.food_pos = [
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1),
        ]
        while self.snake and self.food_pos in self.snake.history:
            self.food_pos = [
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            ]

    def draw_dot(self, pos, color):
        fill(*color)
        ellipse(
            (
                int(
                    pos[0] * self.screen_size / self.size
                    + self.screen_size / self.size / 2
                ),
                int(
                    pos[1] * self.screen_size / self.size
                    + self.screen_size / self.size / 2
                ),
            ),
            int(self.screen_size / self.size),
            int(self.screen_size / self.size),
        )

    def tick(self):
        if self.running:
            self.update()
            if self.show:
                self.draw()

    def update(self):
        self.ticks += 1
        self.snake.feed_forward()
        self.snake.move()
        if self.snake.pos == self.food_pos:
            self.new_food()
            self.fitness += 10
            self.snake.just_ate = True
        # if off screen
        if (
            self.snake.pos[0] < 0
            or self.snake.pos[0] >= self.size
            or self.snake.pos[1] < 0
            or self.snake.pos[1] >= self.size
        ):
            self.running = False
        # if hit self
        if self.snake.pos in self.snake.history:
            self.running = False
        # if alive for too long
        if not (self.show) and self.ticks > max_ticks:
            self.running = False
        # change history history
        self.hishistory.append(self.snake.history)
        del self.hishistory[0]

    def evaluate(self):
        while self.running:
            self.tick()
        return [self.fitness, self.hishistory[0]]

    def draw(self):
        if self.running:
            background(0, 0, 0)
        else:
            background(255, 0, 0)
            print(self.fitness)
        self.snake.draw()
        self.draw_dot(self.food_pos, (123, 1, 2))
        # display snakes certainty
        fill(100, 255, 100)
        rect((20, 20), 50, 300 * self.snake.certain)
