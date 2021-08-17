# -*- coding: utf-8 -*-
"""
Epsilon Greedy strategy
"""
import math 

class Epsilon():
    '''Taken from: https://deeplizard.com/learn/video/PyQNfsGUnQA'''
    def __init__(self,decay,start,end):
        self.decay = decay
        self.start = start
        self.end = end
    def get_exploration_rate(self, steps):
        return self.end + (self.start - self.end) * math.exp(-1. * steps / self.decay)