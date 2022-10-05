from const import *
import random, time
import numpy as np
from Snake import *
from Game import *


def random_population(size):
    population = []
    for i in range(size):
        # make weights
        weights = []
        bias = []
        current_dim = IN_DIM
        for dim in DIMS:
            weights.append(np.random.randn(dim, current_dim))
            bias.append(np.random.randn(dim, 1))
            current_dim = dim
        weights.append(np.random.randn(OUT_DIM, current_dim))
        bias.append(np.random.randn(OUT_DIM, 1))
        population.append([weights, bias])
    return population


games_played = 3


def evaluate(population):
    fitnesses = []
    situations = []
    # set seed to be fair
    global start_seed
    start_seed = [random.random() for i in range(games_played)]
    for i, (weights, bias) in enumerate(population):
        # run a normal game
        print(
            "\rTraining snakes: {0}/{1}, {2}% done".format(
                i, len(population), round(i / len(population) * 100, 1)
            ),
            flush=True,
            end="",
        )
        fitness = 0
        for seed in start_seed:
            game.__init__(GAME_SIZE, weights, bias, start_seed=seed)
            evalu = game.evaluate()
            fitness += evalu[0]
            # get situations
            situations.append(evalu[1])
        fitnesses.append(fitness / games_played)

    # look at a few situations
    situations = situations[: int(len(situations) / 10)]
    for i, (weights, bias) in enumerate(population):
        # run game but with situation
        game.__init__(GAME_SIZE, weights, bias, start_seed=seed)
        game.snake.history = random.choice(situations)
        evalu = game.evaluate()
        fitnesses[i] += evalu[0] * 3

    print("\r{0}/{1}".format(len(population), len(population)), flush=True, end="")
    return fitnesses


def flatten(network):
    ret = []
    for i in range(len(network[0])):
        ret += list(network[0][i].flatten()) + list(network[1][i].flatten())
    return ret


def unflatten(numlist):
    weights = []
    bias = []
    newDIMS = [IN_DIM] + DIMS + [OUT_DIM]
    # del numlist[:newDIMS[0]]
    for i in range(len(newDIMS) - 1):
        dim = newDIMS[i]
        next_dim = newDIMS[i + 1]
        weights.append(np.array(numlist[0 : dim * next_dim]).reshape(next_dim, dim))
        del numlist[0 : dim * next_dim]
        bias.append(np.array(numlist[:next_dim]).reshape(next_dim, 1))
        del numlist[:next_dim]
    return [weights, bias]


def breed(network1, network2):
    flat1 = flatten(network1)
    flat2 = flatten(network2)
    new_flat = []
    split_points = [random.randint(0, len(flat1)) for i in range(random.randint(0, 10))]
    split_points.sort()
    last_point = 0
    i = 0
    for i, point in enumerate(split_points):
        if i % 2 == 0:
            new_flat += flat1[last_point:point]
        else:
            new_flat += flat2[last_point:point]
        last_point = point
    if i % 2 == 0:
        new_flat += flat1[last_point:]
    else:
        new_flat += flat2[last_point:]

    return unflatten(new_flat)


def mutate(network, amount):
    if amount <= 0:
        amount = 1
    flat = flatten(network)
    mutate_points = [random.randint(0, len(flat) - 1) for i in range(amount)]
    for point in mutate_points:
        flat[point] += np.random.randn(1, 1)[0, 0]
    return unflatten(flat)


# def new_population(population, fitnesses, size):
#     sorted_pop = []
#     sorted_fit = []
#     while population:
#         argmax = np.array(fitnesses).argmax()
#         sorted_fit.append(fitnesses[argmax])
#         sorted_pop.append(population[argmax])
#         del fitnesses[argmax]
#         del population[argmax]
#     elite = sorted_pop[:int(len(sorted_pop) * 0.6)]
#     not_elite = [sorted_pop[random.randint(int(len(sorted_pop) * 0.6), len(sorted_pop) - 1)] for i in range(int(len(sorted_pop) * 0.1))]

#     lucky = elite + not_elite
#     needed = size - len(lucky)
#     for i in range(needed):
#         net1 = random.choice(lucky)
#         net2 = random.choice(lucky)
#         lucky.append(breed(net1, net2))
#     lucky = [mutate(net, random.randint(0, 20)) for net in lucky]

#     return lucky
def new_population(population, fitnesses):
    pool = []
    # fitnesses = [f - min(fitnesses) + 1 for f in fitnesses]
    size = len(population)
    for fit, net in zip(fitnesses, population):
        if fit > 0:
            for i in range(round(fit)):
                pool.append(net)
    random.shuffle(pool)
    elite = pool[0 : int(size * 0.75)]
    needed = size - len(elite)
    newPop = elite
    for i in range(needed):
        net1 = random.choice(pool)
        net2 = random.choice(pool)
        newPop.append(breed(net1, net2))
    for i in range(int(size * 0.1)):
        index = random.randint(0, len(newPop) - 1)
        newPop[index] = mutate(
            newPop[index],
            int(100 / (sum(fitnesses) / len(fitnesses)))
            if sum(fitnesses) / len(fitnesses) > 1
            else 150,
        )
    return newPop


game = Game(GAME_SIZE, [], [])
print(
    "Welcome to snake evolution. This is a genetic algorithm that tries to evolve a neural network to play snake."
)
print(
    f"The snake's brain has {IN_DIM} inputs, {OUT_DIM} outputs and hidden layers with dimensions {', '.join([str(n) for n in DIMS])}."
)

start_seed = 1
answer = input("\nWhat should the random seed be? (1)  ")
if answer:
    start_seed = int(answer)


gen_count = 0
print("\nRunning evolution. Press ctrl+c to stop evolution and show results.")
population = random_population(POP_SIZE)
fitnesses = evaluate(population)
print(
    "\rGen: {0}   Max score: {1}   Average score: {2}".format(
        gen_count, max(fitnesses), sum(fitnesses) / len(fitnesses)
    ),
    flush=True,
)
drawGame = Game(GAME_SIZE, [], [])
try:
    while True:
        population = new_population(population, fitnesses)
        fitnesses = evaluate(population)
        print(
            "\rGen: {0}   Max score: {1}   Average score: {2}".format(
                gen_count, max(fitnesses), sum(fitnesses) / len(fitnesses)
            ),
            flush=True,
        )
except:
    pass
print("\nEvolution done! Rendering best snake.")
drawGame.__init__(
    GAME_SIZE,
    *population[fitnesses.index(max(fitnesses))],
    show=True,
    start_seed=random.random(),
)
time.sleep(1)


def setup():
    size(drawGame.screen_size, drawGame.screen_size)


def draw():
    drawGame.tick()


# def key_pressed():
#     if key == 'UP':
#         game.snake.change_direction(1)
#     elif key == 'RIGHT':
#         game.snake.change_direction(2)
#     elif key == 'DOWN':
#         game.snake.change_direction(3)
#     elif key == 'LEFT':
#         game.snake.change_direction(4)

run(frame_rate=20)
