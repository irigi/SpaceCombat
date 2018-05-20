# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:05:03 2018

@author: irigi
"""

import math
import numpy as np
import abc
import pyximport; pyximport.install()
from Physics import ShipCtrl

class DNN(object):
    def __init__(self):
        self.w1 = np.random.rand(7,7)
        self.w2 = np.random.rand(7,7)
        self.w3 = np.random.rand(7,7)
        self.w4 = np.random.rand(3,7)
        
        self.b1 = np.random.rand(7,1)
        self.b2 = np.random.rand(7,1)
        self.b3 = np.random.rand(7,1)
        self.b4 = np.random.rand(3,1)
        
    def sig(self, x):
        x[x > 15] = 15
        x[x < -15] = -15
        
        return 1/(1+np.exp(-x))
        
    def decide(self, inVector):
        v1 = self.sig(self.w1.dot(inVector) - self.b1)
        v2 = self.sig(self.w2.dot(v1) - self.b2)
        v3 = self.sig(self.w3.dot(v2) - self.b3)
        v4 = self.sig(self.w4.dot(v3) - self.b4)
        
        return v4
    
    
class Strategy(metaclass=abc.ABCMeta):    
    def __init__(self):
        self.ctrl = ShipCtrl()
    
    @abc.abstractmethod    
    def get_controls(self, battle):
        return ShipCtrl()
    
    
class StrategyDuck(Strategy):
    def get_controls(self, battle):
        self.ctrl.thr = 0.0
        self.ctrl.eff = 1.0
        
        return self.ctrl
    
    
class StrategyFullThrust(Strategy):
    def get_controls(self, battle):
        self.ctrl.thr = 1.0
        self.ctrl.eff = 0.03
        self.ctrl.phi = 0*np.pi
        
        return self.ctrl
    
    
class StrategyDNN(Strategy):
    def __init__(self):
        Strategy.__init__(self)
        self.dnn = DNN()
        
        # neuron inputs: 
        #    - my       health 
        #    - opponent health
        #    - my       fuel
        #    - opponent fuel
        #    - distance
        #    - velocity towards
        #    - velocity sideways        
    def get_controls(self, battle):
        if battle.shipA.strategy is self:
            myShip = battle.shipA
            opShip = battle.shipB
        else:
            myShip = battle.shipB
            opShip = battle.shipA
            
        deltaX = np.array([opShip.x-myShip.x, opShip.y-myShip.y])
        dist = np.linalg.norm(deltaX)
        dirTowards = deltaX / np.linalg.norm(deltaX)
        dirSideways = np.array([-dirTowards[1], dirTowards[0]])
        
        vel = np.array([opShip.vx-myShip.vx, opShip.vy-myShip.vy])
        vTowards  = np.vdot(vel,dirTowards)
        vSideways = np.vdot(vel,dirSideways)
        
        vecIn = np.transpose(np.matrix(
            [math.log(max(0.1,myShip.health)),
             math.log(max(0.1,opShip.health)),
             math.log(max(0.1,myShip.mFuel)),
             math.log(max(0.1,opShip.mFuel)),
             math.log(max(0.1,dist)),
             vTowards,
             vSideways]))
    
        vecOut = self.dnn.decide(vecIn)

        # neuron outputs: 
        #    - phi
        #    - thr
        #    - eff        
        self.ctrl.phi = vecOut[0]
        self.ctrl.thr = vecOut[1]
        self.ctrl.eff = vecOut[2]
        
        return self.ctrl
