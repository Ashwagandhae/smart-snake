import math
import numpy as np
from const import *


class Snake:
    def __init__(self, brain, start_pos, game, weights, bias):
        self.pos = start_pos
        self.game = game
        # 1 = up
        # 2 = right
        # 3 = down
        # 4 = left
        self.direction = 1
        self.history = [start_pos, start_pos, start_pos, start_pos]
        self.just_ate = False
        self.weights = weights
        self.bias = bias
        self.food_distance = self.get_food_distance()
        self.certain = None

    def draw(self):
        self.game.draw_dot(self.pos, (255, 255, 255))
        for i, pos in enumerate(self.history):
            color = 255
            self.game.draw_dot(pos, (color, color, color))

    def change_direction(self, direction):
        if not (
            (self.direction in (1, 3) and direction in (1, 3))
            or (self.direction in (2, 4) and direction in (2, 4))
        ):
            self.direction = direction

    def get_food_distance(self):
        return (self.pos[0] - self.game.food_pos[0]) ** 2 + (
            self.pos[1] - self.game.food_pos[1]
        ) ** 2

    def move(self):
        self.history.append(self.pos)
        if self.just_ate:
            self.just_ate = False
        else:
            del self.history[0]
        # up
        if self.direction == 1:
            self.pos = [self.pos[0], self.pos[1] - 1]
        # right
        if self.direction == 2:
            self.pos = [self.pos[0] + 1, self.pos[1]]
        # down
        if self.direction == 3:
            self.pos = [self.pos[0], self.pos[1] + 1]
        # left
        if self.direction == 4:
            self.pos = [self.pos[0] - 1, self.pos[1]]

        # calculate fitness
        if self.food_distance > self.get_food_distance():
            self.game.fitness += 1
        else:
            self.game.fitness -= 1 + 2 / math.sqrt(len(self.history))
        self.food_distance = self.get_food_distance()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feed_forward(self):
        # sees three blocks
        # for example
        # block 1, block 2, block 3, block 4, selfx, selfy, foodx, foody
        # 0 empty
        # 1 death
        # 2 food
        # straight, left, right
        death_pos = []
        food_side = []
        snake_side = []
        if self.direction == 1:
            pos_list = [
                [self.pos[0], self.pos[1] - 1],
                [self.pos[0] - 1, self.pos[1]],
                [self.pos[0] + 1, self.pos[1]],
            ]
        elif self.direction == 2:
            pos_list = [
                [self.pos[0] + 1, self.pos[1]],
                [self.pos[0], self.pos[1] - 1],
                [self.pos[0], self.pos[1] + 1],
            ]
        elif self.direction == 3:
            pos_list = [
                [self.pos[0], self.pos[1] + 1],
                [self.pos[0] + 1, self.pos[1]],
                [self.pos[0] - 1, self.pos[1]],
            ]
        elif self.direction == 4:
            pos_list = [
                [self.pos[0] - 1, self.pos[1]],
                [self.pos[0], self.pos[1] + 1],
                [self.pos[0], self.pos[1] - 1],
            ]
        for pos in pos_list:
            # if out of bounds
            if (
                pos[0] < 0
                or pos[0] >= self.game.size
                or pos[1] < 0
                or pos[1] >= self.game.size
            ):
                death_pos.append(1)
            # if self
            elif pos in self.history:
                death_pos.append(1)
            else:
                death_pos.append(0)
        # check if food is in line of sight
        food_up, food_right, food_down, food_left = 0, 0, 0, 0
        if self.pos[1] > self.game.food_pos[1] and self.pos[0] == self.game.food_pos[0]:
            food_up = 1
        if self.pos[0] < self.game.food_pos[0] and self.pos[1] == self.game.food_pos[1]:
            food_right = 1
        if self.pos[1] < self.game.food_pos[1] and self.pos[0] == self.game.food_pos[0]:
            food_down = 1
        if self.pos[0] > self.game.food_pos[0] and self.pos[1] == self.game.food_pos[1]:
            food_left = 1
        # check if snake body is in line of sight
        snake_up, snake_right, snake_down, snake_left = 0, 0, 0, 0
        for pos in self.history:
            if self.pos[1] > pos[1] and self.pos[0] == pos[0]:
                snake_up = 1
            if self.pos[0] < pos[0] and self.pos[1] == pos[1]:
                snake_right = 1
            if self.pos[1] < pos[1] and self.pos[0] == pos[0]:
                snake_down = 1
            if self.pos[0] > pos[0] and self.pos[1] == pos[1]:
                snake_left = 1

        if self.direction == 1:
            # if up
            food_side += [food_up, food_left, food_right]
            snake_side += [snake_up, snake_left, snake_right]

        if self.direction == 2:
            # if right
            food_side += [food_right, food_up, food_down]
            snake_side += [snake_right, snake_up, snake_down]

        if self.direction == 3:
            # if down
            food_side += [food_down, food_right, food_left]
            snake_side += [snake_down, snake_right, snake_left]

        if self.direction == 4:
            # if left
            food_side += [food_left, food_down, food_up]
            snake_side += [snake_left, snake_down, snake_up]

        the_input = np.array(death_pos + food_side + snake_side).reshape(IN_DIM, 1)
        if self.game.show:
            print(the_input)
        # feed forward part
        layer = the_input
        for i in range(len(self.weights)):
            layer = self.sigmoid(np.add(np.dot(self.weights[i], layer), self.bias[i]))
        # do nothing if 0
        if layer.argmax() == 1:
            self.direction -= 1
            if self.direction < 1:
                self.direction = 4
        if layer.argmax() == 2:
            self.direction += 1
            if self.direction > 4:
                self.direction = 1
        # get certainty
        notMax = [layer[i] for i in range(len(layer)) if i != layer.argmax()]
        self.certain = (layer[layer.argmax()] - notMax[0]) + (
            layer[layer.argmax()] - notMax[1]
        ) / 2
