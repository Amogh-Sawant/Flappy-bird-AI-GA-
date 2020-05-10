import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import random 
import pygame
from settings import * 

pygame.init()

class Neural_network:
    def make_model(self):
        self.model = Sequential()
        self.model.add(Dense(4, activation = 'relu', input_shape = (4,), use_bias = False))
        self.model.add(Dense(4, activation = 'relu', use_bias = False))
        self.model.add(Dense(1, activation = 'sigmoid', use_bias = False))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return self.model
    
    def crossover(self, parent1, parent2, mutation):
        for layer in range(len(parent1)):
            for _ in range(random.randrange(len(parent1[layer])*len(parent1[layer][0]))-1):
                self.gene_row = random.randrange(len(parent1[layer]))
                self.gene_column = random.randrange(len(parent1[layer][0]))
                parent1[layer][self.gene_row][self.gene_column] = parent2[layer][self.gene_row][self.gene_column]

        if random.random() <= mutation:
            for layer in range(len(parent1)):
                for _ in range(random.randrange(len(parent1[layer])*len(parent1[layer][0]))-1):
                    self.gene_row = random.randrange(len(parent1[layer]))
                    self.gene_column = random.randrange(len(parent1[layer][0]))
                    parent1[layer][self.gene_row][self.gene_column] = random.uniform(-1.0, 1.0)
        
        return parent1

    def make_crossover_model(self):
        self.model = Sequential()
        self.model.add(Dense(4, activation = 'relu', input_shape = (4,), use_bias = False))
        self.model.add(Dense(4, activation = 'relu', use_bias = False))
        self.model.add(Dense(1, activation = 'sigmoid', use_bias = False))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.model.set_weights(self.crossover(parent1, parent2, 0.3))
        return self.model


class Sprite:
    def __init__(self, y, velocity_y):
        self.y = y
        self.velocity_y = velocity_y

    def draw(self):
        pygame.draw.rect(window, black, [x, self.y, 10, 10])
    
    def new_model(self):
        self.brain = Neural_network()
        self.model_X = self.brain.make_model()
    
    def next_gen_model(self):
        self.brain = Neural_network()
        self.model_X = self.brain.make_crossover_model()

    def same_model(self):
        return self.model_X


class Pipes:
    def __init__(self, x, h):
        pygame.draw.rect(window, black, [x, 0, 10, h])
        pygame.draw.rect(window, black, [x, h+gap, 10, window_len])


def initializing_population():
    global population
    
    for _ in range(population_size):
        population.append(Sprite(y, velocity_y))
    for sprite in population:
        sprite.new_model()

def graphics():
    window.fill(white)

    for sprite in population:
        sprite.draw()

    Pipes(px, h)
    pygame.display.update()
    clock.tick(fps)

def key_strokes():
    global run_game
    for event in pygame.event.get():
        if event.type is pygame.QUIT:
            run_game = False

def collision(y):
    if int(y+20) >= int(window_len) or y <= 0:
        return True
    if x+10 >= px and x <= px+10:
        if y <= h or y >= h+gap:
            return True
    return False

def reset_func():
    global px, y, run_game, x, h, velocity_y, reset, population_size, generation

    x = 100
    y = 100
    px = window_wd
    h = random.randint(20, window_len-gap-20)
    velocity_y = -2     # resets the acceleration
    reset = True
    population_size = 30
    new_generation()
    print("Generation: ", generation)
    generation += 1

def game_logic():
    global px, h

    px -= velocity_x
    if px+10 <= 0:  #infinite random pipe generation
        px = window_wd
        h = random.randint(20, window_len-gap-20)

def distance(a, b):
    return ((a**2) + (b**2))**0.5

def new_generation():
    global population
    for _ in range(population_size):
        population.append(Sprite(y, velocity_y))
    for sprite in population:
        sprite.next_gen_model()

def birds():
    global state
    for sprite in population:
        x1 = distance(abs(sprite.y-h), abs(px-x))  
        x2 = distance(abs(sprite.y-(h+gap)), abs(px-x))
        x3 = sprite.y
        x4 = abs(window_len-sprite.y)
        state = [x1, x2, x3, x4]
        state = [np.array(state)/window_wd]
        state = np.vstack(state)
        brain_model = sprite.same_model()
        jump_probability = (brain_model.predict(state) > 0.5).astype(int)
        # print(brain_model.get_weights())

        if jump_probability[0][0] == 1:
            sprite.velocity_y = 8   # jump
        
        sprite.velocity_y -= 0.4
        sprite.y -= sprite.velocity_y
        
        if collision(sprite.y):
            population.pop(population.index(sprite))

def best_birds():
    global parent1, parent2, best_generation_score, current_generation_score, reset, temp_parent1, temp_parent2

    if len(population) == 2:
        temp_parent1 = (population[0].same_model()).get_weights()
        temp_parent2 = (population[1].same_model()).get_weights()

    if reset:
        reset = False
        if best_generation_score <= current_generation_score:
            best_generation_score = current_generation_score
            parent1 = temp_parent1
            parent2 = temp_parent2
        print("Current score: ", current_generation_score)
        print("best score: ", best_generation_score)
        current_generation_score = 0

    current_generation_score += 1

def game():
    initializing_population()
    while run_game:
        if len(population) == 0:    # basically restarts the game
            reset_func()
        key_strokes()
        birds()
        best_birds()
        game_logic()
        graphics()

if __name__ == '__main__':
    game()

pygame.quit()
quit()


























# class A:
#     def test1(self):
#         self.a = random.random()
#         print(self.a)

#     def test2(self):
#         return (self.a)

# b = A()
# b1 = A()
# b.test1()
# b1.test1()
# x1 = b.test2()
# x2 = b1.test2()
# x = b.test2()

# print(x1)
# print(x2)
# print(x)