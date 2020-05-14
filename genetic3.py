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
        self.model.add(Dense(8, activation = 'relu', use_bias = False))
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
        self.model.add(Dense(8, activation = 'relu', use_bias = False))
        self.model.add(Dense(1, activation = 'sigmoid', use_bias = False))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.model.set_weights(self.crossover(parent1 = parent1, parent2 = parent2, mutation = 0.35))
        return self.model


class Sprite:

    def __init__(self):
        self.x = 100
        self.y = 300
        self.velocity_y = 0
        self.gravity = 0.4

    def draw(self):
        pygame.draw.rect(window, black, [self.x, self.y, 10, 10])

    def free_fall(self):
        self.velocity_y -= self.gravity
        self.y -= self.velocity_y

    def collision(self, pipe):
        if int(self.y+10) >= int(window_len) or self.y <= 0:
            return True
        if self.x+10 >= pipe.x and self.x <= pipe.x+10:
            if self.y <= pipe.h or self.y >= pipe.h+gap:
                return True
        return False

    def jump(self):
        self.velocity_y = 8
    
    def new_model(self):
        self.brain = Neural_network()
        self.model_X = self.brain.make_model()
    
    def next_gen_model(self):
        self.brain = Neural_network()
        self.model_X = self.brain.make_crossover_model()

    def same_model(self):
        return self.model_X


class Pipes:

    def __init__(self):
        self.x = window_wd
        self.h = random.randint(20, window_len-gap-20)
        self.velocity_x = 2 
    
    def draw(self):
        pygame.draw.rect(window, black, [self.x, 0, 10, self.h])
        pygame.draw.rect(window, black, [self.x, self.h+gap, 10, window_len])

    def procedural_generation(self):
        self.x -= self.velocity_x
        if self.x + 10 <= 0: 
            self.x = window_wd
            self.h =  random.randint(20, window_len-gap-20)


def key_strokes():
    global run_game

    for event in pygame.event.get():
        if event.type is pygame.QUIT:
            run_game = False

def initializing_population():
    global population

    for _ in range(population_size):
        population.append(Sprite())
    for sprite in population:
        sprite.new_model()

def new_generation():
    global population

    for _ in range(population_size):
        population.append(Sprite())
    for sprite in population:
        sprite.next_gen_model()

def graphics(pipe):
    window.fill(white)
    for sprite in population:
        sprite.draw()

    pipe.draw()
    pygame.display.update()
    clock.tick(fps)

def reset_func(pipe):
    global run_game, reset, population_size, generation

    pipe.x = window_wd
    pipe.h = random.randint(20, window_len-gap-20)
    reset = True
    population_size = 30
    new_generation()
    print("Generation: ", generation)
    generation += 1

def distance(a, b):
    return ((a**2) + (b**2))**0.5

def birds(pipe):
    global state

    for sprite in population:
        x1 = distance(abs(sprite.y-pipe.h), abs(pipe.x-sprite.x))  
        x2 = distance(abs(sprite.y-(pipe.h+gap)), abs(pipe.x-sprite.x))
        x3 = sprite.y
        x4 = abs(window_len-sprite.y)
        state = [x1, x2, x3, x4]
        state = [np.array(state)/window_wd]
        state = np.vstack(state)
        brain_model = sprite.same_model()
        jump_probability = (brain_model.predict(state) > 0.5).astype(int)

        if jump_probability[0][0] == 1:
            sprite.jump()
        
        sprite.free_fall()
        if sprite.collision(pipe):
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
    pipe = Pipes()
    while run_game:
        if len(population) == 0:    # basically restarts the game
            reset_func(pipe)
        key_strokes()
        birds(pipe)
        best_birds()
        pipe.procedural_generation()
        graphics(pipe)

if __name__ == '__main__':
    game()

pygame.quit()
quit()