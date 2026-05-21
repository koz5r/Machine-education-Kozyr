# x² + y² + z² = 1000
# x + y + z = 50

import numpy as np
import random
import matplotlib.pyplot as plt

class GA:
    def __init__(self, pop_size=500, gens=200, mut_rate=0.15, cross_rate=0.85):
        self.pop_size = pop_size
        self.gens = gens
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.bounds = (-100, 100)
        self.best_sol = None
        self.best_err = float('inf')
        self.history = []
    
    def fitness(self, x, y, z):
        eq1 = x**2 + y**2 + z**2 - 1000
        eq2 = x + y + z - 50
        return abs(eq1) + abs(eq2)
    
    def create(self):
        return [random.randint(self.bounds[0], self.bounds[1]) for _ in range(3)]
    
    def population(self):
        return [self.create() for _ in range(self.pop_size)]
    
    def select(self, pop, fits):
        selected = []
        for _ in range(len(pop)):
            idx = random.sample(range(len(pop)), 3)
            winner = idx[np.argmin([fits[i] for i in idx])]
            selected.append(pop[winner].copy())
        return selected
    
    def crossover(self, p1, p2):
        if random.random() < self.cross_rate:
            point = random.randint(1, 2)
            c1 = p1[:point] + p2[point:]
            c2 = p2[:point] + p1[point:]
            return c1, c2
        return p1.copy(), p2.copy()
    
    def mutate(self, ind):
        for i in range(3):
            if random.random() < self.mut_rate:
                ind[i] = random.randint(self.bounds[0], self.bounds[1])
        return ind
    
    def solve(self):
        pop = self.population()
        
        for gen in range(self.gens):
            fits = [self.fitness(ind[0], ind[1], ind[2]) for ind in pop]
            
            best_idx = np.argmin(fits)
            if fits[best_idx] < self.best_err:
                self.best_err = fits[best_idx]
                self.best_sol = pop[best_idx].copy()
            
            self.history.append(self.best_err)
            
            if self.best_err < 0.1:
                break
            
            pop = self.select(pop, fits)
            
            new_pop = []
            for i in range(0, len(pop), 2):
                if i + 1 < len(pop):
                    c1, c2 = self.crossover(pop[i], pop[i+1])
                    new_pop.append(c1)
                    new_pop.append(c2)
                else:
                    new_pop.append(pop[i].copy())
            
            pop = [self.mutate(ind) for ind in new_pop]
        
        return self.best_sol, self.best_err
    
    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, linewidth=2, color='darkgreen')
        plt.xlabel('Generation')
        plt.ylabel('Error')
        plt.title('Convergence')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    ga = GA(pop_size=500, gens=200, mut_rate=0.15, cross_rate=0.85)
    
    solution, error = ga.solve()
    
    x, y, z = solution
    print(f"Solution: ({x}, {y}, {z})")
    print(f"x²+y²+z² = {x**2 + y**2 + z**2} (need 1000)")
    print(f"x+y+z = {x + y + z} (need 50)")
    print(f"Error: {error}")
    
    ga.plot()