# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pyximport; pyximport.install()
from Strategy import StrategyDNN, StrategyDuck
from Physics import Ship

spread = math.cos(15.0 / 180 * np.pi)
targetSize = 100000
startDst = 1.5e5
dt = 1
tMax = 3600
healthMaxAttacker = 20
healthMaxDefender = 15


        
class Battle(object):
    def __init__(self, shipA, shipB):
        self.shipA = copy.deepcopy(shipA)
        self.shipB = copy.deepcopy(shipB)
        
        self.shipA.healthMax = healthMaxAttacker
        self.shipB.healthMax = healthMaxDefender
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
        
        if math.cos(ctrlA.phi) > spread:
            if random.random() < targetSize / dstSq:
                self.shipB.health = self.shipB.health - 1
        
        
        ctrlB = self.shipB.strategy.get_controls(battle=self)

        phi = phi + np.pi  #math.atan2(self.shipA.y-self.shipB.y, self.shipA.x-self.shipB.x)
        c = math.cos(phi+ctrlB.phi)
        s = math.sin(phi+ctrlB.phi)
        
        FB = self.shipB.F(ctrlB)
        FxB = c * FB
        FyB = s * FB
        
        self.shipB.propagate(Fx=FxB,Fy=FyB,dt=dt,ctrl=ctrlB)
        
        if math.cos(ctrlB.phi) > spread:
            if random.random() < targetSize / dstSq:
                self.shipA.health = self.shipA.health - 1        
        
    def score(self, sA, sB):
        '''
        score for undecided battles, attains values [-1,1]
        1 = shipA victory, -1 = shipB victory
        '''
        
        if self.sA.health <= 0.0:
            if self.sB.health <= 0.0:
                return 0
            else:
                return -1
            
        if self.sB.health <= 0.0:
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


    
      
def printStats():
    b = Battle(shipA=Ship(strategy=StrategyDNN()), 
               shipB=Ship(strategy=StrategyDuck())) 
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
    
    #plt.figure(2)
    #plt.plot(tt,FA)
    #plt.show()
    
    #plt.figure(3)
    #plt.plot(tt,mA)
    #plt.show()        

    #plt.figure(4)
    #plt.plot(tt,hA)
    #plt.plot(tt,hB)
    #plt.show()

    
if __name__ == "__main__":
    
    #res = {}
    #res[-1] = 0
    #res[0] = 0
    #res[1] = 0
    #for i in range(0,30):
    #    b = Battle(shipA=Ship(strategy=StrategyFullThrust(), healthMax = 12.0), 
    #               shipB=Ship(strategy=StrategyFullThrust(), healthMax = 10.0)
    #               )
    #    result = b.play_battle()
    #    
    #    res[result] = res[result] + 1
    #    
    #print (res)
    
    #dn = DNN()
    #dn.decide(np.transpose(np.matrix([1,1,1,1,1,1,1])))
    
    printStats()