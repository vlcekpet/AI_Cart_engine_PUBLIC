from numpy import random as np_random
import random
import numpy as np
import copy
import string
from AI_engines.ShallowNNRandom import ShallowNNRandom

class AIbrain_TeamName:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        self.decider = ShallowNNRandom(n_inputs=9, n_hidden=10, n_outputs=4, weight_scale=0.5)

        self.init_param()

    def init_param(self):
        self.NAME ="Safr_"+''.join(random.choices(self.chars, k=5))
        self.store()

    def store(self):
        self.parameters = copy.deepcopy({
            'w1': self.decider.W1,
            'w2': self.decider.W2,
            'b1': self.decider.b1,
            'b2': self.decider.b2,
            "NAME": self.NAME,
        })
        print(f"Brain stored: {self.parameters}")

    def decide(self, data):
        return self.decider.predict_proba(data)

    def mutate(self):
        noise_scale = 0.9
        change_chance = 0.75
        for i in self.decider.W1:
            if np_random.rand(1) < change_chance:
                i += np_random.choice([-1, 1]) * noise_scale * i
        self.NAME += "_MUT_W1_"+''.join(random.choices(self.chars, k=3))
    
        for i in self.decider.W2:
            if np_random.rand(1) < change_chance:
                i += np_random.choice([-1, 1]) * noise_scale * i
        self.NAME += "_MUT_W2_"+''.join(random.choices(self.chars, k=3))

        for b in self.decider.b1:
            if np_random.rand(1) < change_chance:
                b += np_random.choice([-1, 1]) * 0.2 * b
        for b in self.decider.b2:
            if np_random.rand(1) < change_chance:
                b += np_random.choice([-1, 1]) * 0.2 * b
        self.store()

    def calculate_score(self, distance, time, no):
        self.score = distance**2/time
        print(f"Brain {self.NAME} score calculated: {self.score}")

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            self.parameters = copy.deepcopy(parameters)

        self.decider.W1 = self.parameters['w1']
        self.decider.W2 = self.parameters['w2']
        self.decider.b1 = self.parameters['b1']
        self.decider.b2 = self.parameters['b2']
        self.NAME = self.parameters['NAME']
