# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import copy
import abc
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pyximport; pyximport.install()

spread = math.cos(15.0 / 180 * np.pi)
targetSize = 100000
startDst = 1.5e5
dt = 1
tMax = 3600

class SpaceObject(object):
    def __init__(self, x = 0.0, y = 0.0, m = 1.0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        
        self.__m = m                # not to be reused in child classes
        
    def mass(self):
        return self.__m
    
    def propagate(self, Fx, Fy, dt):        
        self.__propagate_Exact(Fx=Fx, Fy=Fy, dt=dt)
    
    def __propagate_Exact(self, Fx, Fy, dt):
        x0 = self.x
        vx0 = self.vx
        y0 = self.y
        vy0 = self.vy
        m0 = self.mass()
        
        
        self.x = x0 + vx0 * dt + Fx * dt*dt / (2 * m0)
        self.vx = vx0 + Fx * dt / m0
        
        self.y = y0 + vy0 * dt + Fy * dt*dt / (2 * m0) 
        self.vy = vy0 + Fy * dt / m0

    def __propagate_Euler(self, Fx, Fy, dt):
        m = self.mass()
        
        self.x = self.x  + self.vx * dt
        self.vx = self.vx  + Fx / m * dt

        self.y = self.y  + self.vy * dt
        self.vy = self.vy  + Fy / m * dt        


class ShipCtrl(object):
    def __init__(self):
        self.phi = 0.0
        self.thr = 0.0
        self.eff = 0.0
        
    def correct(self):
        self.thr = max(0.0,min(1.0,self.thr))
        self.eff = max(0.0,min(1.0,self.eff))


class Ship(SpaceObject):
    def __init__(self, strategy, vOut = 11000000.0, pwrGW = 5500.0,
                 mFuelMax = 300000.0, mPayload = 100000.0,
                 healthMax = 10.0):
        
        SpaceObject.__init__(self, m = mFuelMax + mPayload)
        
        self.vOut = vOut
        self.pwrGW = pwrGW
        self.mFuel = mFuelMax
        self.mFuelMax = mFuelMax
        self.mPayload = mPayload
        self.health = healthMax
        self.healthMax = healthMax
        self.strategy = strategy
        
    def mass(self):
        return self.mPayload + self.mFuel
        
    def F(self, ctrl):        
        if self.mFuel <= 0:
            return 0.0
        
        eff = max(0.0,min(1.0,ctrl.eff))
        return 2 * self.pwrGW / (self.vOut * eff) * 1e9
    
    def dm(self, ctrl):        
        if self.mFuel <= 0:
            return 0.0
        
        eff = max(0.0,min(1.0,ctrl.eff))
        return 2 * self.pwrGW / ((self.vOut*eff)**2) * 1e9
    
    
    def propagate(self, Fx, Fy, dt, ctrl):
        self.__propagate_Exact(Fx, Fy, dt, ctrl)
        
    def __propagate_Euler(self, Fx, Fy, dt, ctrl):
        #self.mFuel = max(0.0,self.mFuel - self.dm(ctrl)*dt/2)
        SpaceObject.propagate(self, Fx=Fx, Fy=Fy, dt=dt)
        self.mFuel = max(0.0,self.mFuel - self.dm(ctrl)*dt)
        
    def __propagate_Exact(self, Fx, Fy, dt, ctrl):
        if self.mFuel <= 0.0:
            SpaceObject.propagate(self, Fx=0,Fy=0,dt=dt)
            
        else:
            dm = self.dm(ctrl)
            m = self.mass()
            x0 = self.x
            vx0 = self.vx
            y0 = self.y
            vy0 = self.vy
        
            tFuel = min(dt,self.mFuel / dm*dt)
            tEmpty = dt - tFuel
        
            if tFuel > 1e-8:
                self.mFuel = max(0.0,self.mFuel - dm*tFuel)
                self.x = (dm*(Fx*tFuel + dm*(tFuel*vx0 + x0)) + 
                  Fx*(m - dm*tFuel)*math.log((m - dm*tFuel)/m))/(dm**2)
        
                self.vx = (-(dm*Fx) + dm*(Fx + dm*vx0) 
                  - dm*Fx*math.log((m - dm*tFuel)/m))/(dm**2)
        
                self.y = (dm*(Fy*tFuel + dm*(tFuel*vy0 + y0)) + 
                  Fy*(m - dm*tFuel)*math.log((m - dm*tFuel)/m))/(dm**2)
        
                self.vy = (-(dm*Fy) + dm*(Fy + dm*vy0) 
                  - dm*Fy*math.log((m - dm*tFuel)/m))/(dm**2)
      
            if tEmpty > 1e-8:
                SpaceObject.propagate(self, Fx=0,Fy=0,dt=tEmpty)
            

        
class Battle(object):
    def __init__(self, shipA, shipB):
        self.shipA = copy.deepcopy(shipA)
        self.shipB = copy.deepcopy(shipB)
                
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
        
    def play_battle(self):
        t = 0.0
        
        while t < tMax:
            t = t + dt
            
            if self.shipA.health <= 0.0:
                return -1
            
            if self.shipB.health <= 0.0:
                return 1
        
            self.play_turn()
        
        return 0

           
# neuron outputs: 
#    - phi
#    - thr
#    - eff
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
            
        
        vecIn = np.transpose(np.matrix(
            [math.log(max(0.1,myShip.health)),
             math.log(max(0.1,opShip.health)),
             math.log(max(0.1,myShip.mFuel)),
             math.log(max(0.1,opShip.mFuel)),
             math.log(max(0.1,(myShip.x-opShip.x)**2+(myShip.y-opShip.y)**2))/2,
             math.log(2),
             math.log(2)]))
    
        self.ctrl.eff = 1.0
        return self.ctrl
    
      
def printStats():
    b = Battle(shipA=Ship(strategy=StrategyFullThrust()), 
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