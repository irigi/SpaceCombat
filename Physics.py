# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:09:20 2018

@author: irigi
"""

import math
import pyximport; pyximport.install()


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
