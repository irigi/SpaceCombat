# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pyximport; pyximport.install()
from Strategy import StrategyDNN, StrategyDuck, StrategyFullThrust, StrategyFullDuck
from Physics import Ship

spread = math.cos(15.0 / 180 * np.pi)
targetSize = 100000*100
startDst = 1.5e5
dt = 5
tMax = 3600
healthMaxAttacker = 20
healthMaxDefender = 15


        
class Battle(object):
    def __init__(self, strategyA, strategyB):
        self.shipA = Ship(strategy=strategyA, healthMax=healthMaxAttacker)
        self.shipB = Ship(strategy=strategyB, healthMax=healthMaxDefender)
        
        self.reset_ships()
        
    def reset_ships(self):        
        self.shipA.health = self.shipA.healthMax
        self.shipB.health = self.shipB.healthMax
        
        self.shipA.y = -startDst/2
        self.shipA.x = 0.0
        self.shipA.vx = 0.0
        self.shipA.vy = 0.0
        
        self.shipB.y = startDst/2
        self.shipB.x = 0.0
        self.shipB.vx = 0.0
        self.shipB.vy = 0.0
    
    def play_turn(self):
        dstSq = (self.shipB.y-self.shipA.y)**2 + (self.shipB.x-self.shipA.x)**2        
        
        ctrlA = self.shipA.strategy.get_controls(battle=self)

        phi = math.atan2(self.shipB.y-self.shipA.y, self.shipB.x-self.shipA.x)
        c = math.cos(phi+ctrlA.phi)
        s = math.sin(phi+ctrlA.phi)
        
        FA = self.shipA.F(ctrlA)
        FxA = c * FA
        FyA = s * FA
        
        self.shipA.propagate(Fx=FxA, Fy=FyA, dt=dt, ctrl=ctrlA)
        
        ctrlB = self.shipB.strategy.get_controls(battle=self)

        phi = phi + np.pi  #math.atan2(self.shipA.y-self.shipB.y, self.shipA.x-self.shipB.x)
        c = math.cos(phi+ctrlB.phi)
        s = math.sin(phi+ctrlB.phi)
        
        FB = self.shipB.F(ctrlB)
        FxB = c * FB
        FyB = s * FB
        
        self.shipB.propagate(Fx=FxB, Fy=FyB, dt=dt, ctrl=ctrlB)

        hitProbability = min(1,targetSize / dstSq)
        if math.cos(ctrlA.phi) > spread:
            self.shipB.health = self.shipB.health - hitProbability # to reduce noise during learning 
            
            #if random.random() < hitProbability:
            #    self.shipB.health = self.shipB.health - 1
        
        if math.cos(ctrlB.phi) > spread:
            self.shipA.health = self.shipA.health - hitProbability # to reduce noise during learning 
            
            #if random.random() < targetSize / dstSq:
            #    self.shipA.health = self.shipA.health - 1        
        
    def score(self, sA, sB):
        '''
        score for undecided battles, attains values [-1,1]
        1 = shipA victory, -1 = shipB victory
        '''
        
        if sA.health <= 0.0:
            if sB.health <= 0.0:
                return 0
            else:
                return -1
            
        if sB.health <= 0.0:
            return 1        

        return sA.health/sA.healthMax - sB.health/sB.healthMax
        
    def play_battle(self):        
        self.reset_ships()

        t = 0.0        
        while t < tMax:
            t = t + dt
            
            if self.shipA.health <= 0.0:
                break
            
            if self.shipB.health <= 0.0:
                break
        
            self.play_turn()
        
        return self.score(self.shipA, self.shipB)


class BreedingPool(object):
    def __init__(self):
        self.NNN = 2
        self.pool = []
        for i in range(0,self.NNN):
            if i == 0:
                strategy = StrategyFullThrust()
            else:
                strategy = StrategyFullDuck()

            strategy.bp_score = 0
            self.pool.append(strategy)
        
    def play(self):
        for rd in range(0,100):
            for strat in self.pool:
                #rnd = np.random.randint(self.NNN)
                for strat2 in self.pool:
                    result = Battle(strat, strat2).play_battle()
                    strat.bp_score = strat.bp_score + result
                    strat2.bp_score = strat2.bp_score - result
                
            print ([a.bp_score for a in self.pool])
    
    
    
      
def printStats():
    b = Battle(StrategyFullThrust(), 
               StrategyDuck() )
    tt = []
    xA = []
    yA = []
    FA = []
    mA = []
    hA = []
    hB = []
    t = 0
    while t < tMax:
        tt.append(t)
        xA.append(b.shipA.x)
        yA.append(b.shipA.y)
        FA.append(b.shipA.F(b.shipA.strategy.get_controls(b)))
        mA.append(b.shipA.mass()) 
        
        hA.append(b.shipA.health)
        hB.append(b.shipB.health)
        
        b.play_turn()
        t = t + dt

    plt.figure(1)    
    plt.plot(tt,yA)
    plt.show()
    
    #print(len(tt), len(yA))
    
    plt.figure(2)
    plt.plot(tt,hA)

    
    plt.figure(2)
    plt.plot(tt,hB)
    plt.show()
    
    #plt.figure(3)
    #plt.plot(tt,mA)
    #plt.show()        

    #plt.figure(4)
    #plt.plot(tt,hA)
    #plt.plot(tt,hB)
    #plt.show()

    
if __name__ == "__main__":
    
    poolTmp = BreedingPool()
    poolTmp.play()
    
    #printStats()