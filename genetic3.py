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
        self.model.add(Dense(8, activation = 'relu', input_shape = (4,), use_bias = False))
        self.model.add(Dense(16, activation = 'relu', use_bias = False))
        self.model.add(Dense(1, activation = 'sigmoid', use_bias = False))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return self.model
    
    def crossover(self, parent_1, parent_2, mutation):
        self.child = [[[None for _ in range(8)] for _ in range(4)], [[None for _ in range(16)] for _ in range(8)], [[None for _ in range(1)] for _ in range(16)]]

        for layer in range(len(self.child)):
            for gene_row in range(len(self.child[layer])):
                for gene_col in range(len(self.child[layer][0])):
                    if random.choice(("parent1", "parent2")) is "parent1":
                        self.child[layer][gene_row][gene_col] = parent_1[layer][gene_row][gene_col]
                    else:
                        self.child[layer][gene_row][gene_col] = parent_2[layer][gene_row][gene_col]

        if random.random() <= mutation:
            for layer in range(len(parent_1)):
                for _ in range(3, (random.randrange(len(parent_1[layer])*len(parent_1[layer][0]))-1)):
                    self.gene_row = random.randrange(len(parent_1[layer]))
                    self.gene_column = random.randrange(len(parent_1[layer][0]))
                    parent_1[layer][self.gene_row][self.gene_column] = random.uniform(-1.0, 1.0)
        
        return parent_1

    def make_crossover_model(self):
        self.model = Sequential()
        self.model.add(Dense(8, activation = 'relu', input_shape = (4,), use_bias = False))
        self.model.add(Dense(16, activation = 'relu', use_bias = False))
        self.model.add(Dense(1, activation = 'sigmoid', use_bias = False))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.model.set_weights(self.crossover(parent_1 = parent1, parent_2 = parent2, mutation = 0.3))
        return self.model


class Sprite:

    def __init__(self):
        self.x = 100
        self.y = 300
        self.velocity_y = 0
        self.gravity = 0.4
        self.fitness_score = 0

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

    def distance(self, a, b):
        return ((a**2) + (b**2))**0.5

    def calculating_fitness_score(self, pipe):
        if self.x >= pipe.x:
            self.fitness_score += 200
        else:
            self.fitness_score += 1/(self.distance(abs(self.y-(pipe.h+int(gap/2))), abs(pipe.x-self.x)))


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
            if len(population) != 0:
                for sprite in population:
                    fitness_score_array.append([sprite.same_model(), sprite.fitness_score])
                    population.pop(population.index(sprite))
            best_birds()

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

def distance(a, b):
    return ((a**2) + (b**2))**0.5

def birds(pipe):
    global state

    for sprite in population:
        sprite.free_fall()
        sprite.calculating_fitness_score(pipe)

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
        
        if sprite.collision(pipe):
            fitness_score_array.append([brain_model, sprite.fitness_score])
            population.pop(population.index(sprite))

def best_birds():
    global fitness_score_array, parent1, parent2, reset
 
    fitness_score_array = np.array(fitness_score_array)
    fitness_score_array = fitness_score_array[fitness_score_array[:, 1].argsort()] 
    
    parent1 = fitness_score_array[len(fitness_score_array)-1][0].get_weights()
    parent2 = fitness_score_array[len(fitness_score_array)-2][0].get_weights()
    save_model()
    fitness_score_array = []

def reset_func(pipe):
    global run_game, reset, population_size, generation

    pipe.x = window_wd
    pipe.h = random.randint(20, window_len-gap-20)
    reset = True
    population_size = 30
    best_birds()
    new_generation()
    print("Generation: ", generation)
    generation += 1

def save_model():
    global current_generation_score, best_generation_score

    current_generation_score = fitness_score_array[len(fitness_score_array)-1][1]
    if current_generation_score >= best_generation_score:
        best_generation_score = current_generation_score
        model = fitness_score_array[len(fitness_score_array)-1][0]
        model.save('GA-AI3.h5')

def game():
    initializing_population()
    pipe = Pipes()
    while run_game:
        if len(population) == 0:    # basically restarts the game
            reset_func(pipe)
        key_strokes()
        birds(pipe)
        pipe.procedural_generation()
        graphics(pipe)

if __name__ == '__main__':
    game()

pygame.quit()
quit()
