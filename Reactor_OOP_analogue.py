#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:55:05 2020

@author: Hanshen Su
"""

import numpy.random as rnd
import numpy as np

import scipy
from scipy import constants

import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.patches import Rectangle

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

import os
from pprint import pprint
import shutil
import subprocess
import urllib.request

import h5py

import openmc.data

import time

import copy

import pickle

#%%
colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")

""" Define Constants"""
neuron_mass = constants.neutron_mass
uranium_density = 19.1
avogadro =constants.Avogadro



initial_energy = 1000000

# = 0.08


#%%

# import data from library


u235 = openmc.data.IncidentNeutron.from_endf("/Users/a1234/Database/ENDF-B-VIII.0/neutrons/n-092_U_235.endf")
u238 = openmc.data.IncidentNeutron.from_endf("/Users/a1234/Database/ENDF-B-VIII.0/neutrons/n-092_U_238.endf")
o16 = openmc.data.IncidentNeutron.from_endf("/Users/a1234/Database/ENDF-B-VIII.0/neutrons/n-008_O_016.endf")
""" Define Classes """

""" Medium """
class Medium:

    atoms = np.empty((0,4))
    density = 0
    def __init__(self, density, atoms):
        """
        atoms example
        atoms = [[U235,0.6,1],[U238,0.4,1],[O16,1,2]]
        [atomtype,percentage,number of atoms per molecule]
        """
        self.density = density
        self.atoms = np.array(atoms)
    def name(self):
        return self.atoms[0]
    def totalmass(self):
        totalmass = 0
        for i in self.atoms:
            totalmass += i[0].mass * i[1] * i[2]
        return totalmass
    def totalMXsection(self, energy):
        number_of_atoms = avogadro * self.density / self.totalmass() *1E6# cm to m
        totalcrosssection = 0
        for i in self.atoms:
            totalcrosssection += i[0].totalXsection(energy)*number_of_atoms*i[1]*10**(-28)*i[2]
        return totalcrosssection
    def partialXsection(self, energy):
        pxs = np.empty((0,2))
        number_of_atoms = avogadro * self.density / self.totalmass() *1E6#cm to m
        for i in self.atoms:
            temp = [i[0] ,i[0].totalXsection(energy)*number_of_atoms*i[1] * i[2] *10**(-28)]
            pxs = np.vstack((pxs,temp))
        return pxs
    def length_distribution(self, energy):
        mean_freepath = 1/self.totalMXsection(energy)
        return -mean_freepath*np.log(rnd.uniform(0,1))
    def dice_atom(self, energy):
        """
        This is a random process
        """
        atoms = self.partialXsection(energy)[:,0]
        pxs = self.partialXsection(energy)[:,1].astype(float)
        prob = pxs/np.sum(pxs)
        return rnd.choice(atoms,1,p=prob)[0]

""" Atom """
class Atom:
    def __init__(self, endf):
        self.endf = endf
    def name(self):
        return self.endf.name
    def elastic(self, energy):
        return self.endf[2].xs['0K'](energy)
    def elastic_angle(self, direction):
        x = direction[0]
        y = direction[1]
        z = direction[2]
        coscm = rnd.uniform(-1,1)
        A = self.mass   # note that self.mass must be defined outside manually
        coslab = (1 + A*coscm) * (1+A**2+2*A*coscm)**(-1/2)
        chi = rnd.uniform(0,1) * 2*np.pi
        cosx = np.cos(chi)
        sinx = np.sin(chi)
        a = (1-z**2)**(1/2)
        z1 = z*coslab + a*cosx
        b = coslab - z*z1
        y1 = 1/(1-z**2) * (y*b + x*a*sinx)
        x1 = 1/(1-z**2) * (x*b - y*a*sinx)
        return np.array([x1,y1,z1])/np.linalg.norm([x1,y1,z1])
    def elastic_energy(self, energy):
        A = self.mass
        a = (A-1)/(A+1)
        return 1/2*energy*((1-a**2)*rnd.uniform(-1,1)+1+a**2)
    def level(self, energy):
        return self.endf[4].xs['0K'](energy)
    def Q(self):
        return self.endf[4].q_value
    def inelastic_angle_plus(self,vector,energy):
        x = vector[0]
        y = vector[1]
        z = vector[2]
        coscm = rnd.uniform(-1,1)
        A = self.mass
        Q = self.Q()
        g = (A**2 + A*(A+1)*Q/energy)**(-1/2)
        coslab = (g + coscm) * (1+g**2+2*g*coscm)**(-1/2)
        chi = rnd.uniform(0,1) * 2*np.pi
        cosx = np.cos(chi)
        sinx = np.sin(chi)
        a = (1-z**2)**(1/2)
        z1 = z*coslab + a*cosx
        b = coslab - z*z1
        y1 = 1/(1-z**2) * (y*b + x*a*sinx)
        x1 = 1/(1-z**2) * (x*b - y*a*sinx)
        return np.array([x1,y1,z1])/np.linalg.norm([x1,y1,z1]),(A+1)**(-2)*(coslab*(energy)**(1/2)+\
                 (energy*(coslab**2+A**2-1) + A*(A+1)*Q)**(1/2))**2 
    def fission(self, energy):
        if self.name() == 'O16':
            return 0
        else:
            return self.endf[18].xs['0K'](energy)
    def ngamma(self, energy):
        return self.endf[102].xs['0K'](energy)
    def totalXsection(self,energy):
        totalXS = self.elastic(energy) \
                 +self.level(energy) \
                 +self.fission(energy) \
                 +self.ngamma(energy)
        return totalXS

    def dice_reaction(self, energy):
        """
        This is a random process
        """
        outcome = ["elastic","level","fission","ngamma"]
        prob = np.array([self.elastic(energy),
                         self.level(energy),
                         self.fission(energy),
                         self.ngamma(energy)])/self.totalXsection(energy)
        return rnd.choice(outcome,1,p=prob)[0]
    def u235_fission_new(self, position, energy):
        """
        This is a random process
        """
        avg = 2.43# avg neutrons realeased per fission
        prob = 3 - avg
        release = rnd.choice([2,3],p=([prob,1-prob]))
        neutron_seq = np.array([])
        for i in range(release):
            neutron = Neutron(position, watts_distribution(), unit_vector())
            neutron_seq = np.hstack((neutron_seq,neutron))
        return neutron_seq

""" Neutron """
class Neutron:
    """this is a docstring"""
    alive = True
    # instance attributes
    def __init__(self, position, energy, direction):
        self.position = position
        self.energy = energy
        self.direction = direction
    # instance methods
    def move(self, medium):
        """
        TODO
        Let this depend on the medium 
        Done
        This is a random process
        """
        move = self.direction * medium.length_distribution(self.energy)
        self.position = self.position + move
        
    def collision(self, medium):
        """
        TODO
        Let this depend on the medium
        This is a random process
        """
        diceA = medium.dice_atom(self.energy)
        diceR = diceA.dice_reaction(self.energy)
        if diceR == "elastic":
                self.energy = diceA.elastic_energy(self.energy)
                self.direction = diceA.elastic_angle(self.direction)
                           
        elif diceR == "level":
            if self.energy > -diceA.Q():
                self.direction, self.energy = diceA.inelastic_angle_plus(self.direction,self.energy)
            else:
                self.alive = False
                #energy = u235_elastic_energy(energy)
                #direction = u235_elastic_angle(direction) 
            
        elif diceR == "fission":
            self.alive = False
            
            self.new_neutrons = diceA.u235_fission_new(self.position, self.energy)
            # Set the new neutrons
            return True
        elif diceR == "ngamma":
            self.alive = False

""" Geometry """
class Geometry:
    """ 
    Define the geometry of the reactor
        shape : TYPE
        "shpere","cylinder","rectangle".
    """  
    cube = 1 # a cube that contains the reactor
    def __init__(self, shape,r=0,x=0,y=0,z=0):
        """
        shape : TYPE
            "shpere","cylinder","rectangle".
        """
        self.shape = shape
        self.radius = r
        self.height = z
        self.length = x
        self.width = y
        if shape == "sphere":
            self.cube = self.radius
        if shape == "cylinder":
            self.cube = max(self.height/2,self.radius)
        if shape =="rectangle":
            self.cube = max(self.height,self.length,self.width)/2
    def inside(self, position):
        
        if self.shape == "sphere":
            if np.linalg.norm(position) <= self.radius:
                return True
            else:
                return False
        if self.shape == "cylinder":
            if np.linalg.norm(position[:2]) <= self.radius \
            and abs(position[2])<=self.height/2:
                return True
            else:
                return False
        if self.shape =="rectangle":
            if abs(position[0]) <= self.length/2\
            and abs(position[1]) <= self.width/2\
            and abs(position[2]) <= self.height/2:
                return True
            else:
                return False
    def spawn_neutron(self):
        while True:
            sample = rnd.uniform(-1,1,size=3)*self.cube
            if self.inside(sample):
                return sample
            else:
                continue 
    def check_medium(self, position):
        if self.inside(position):
            return self.medium
        else:
            return "out"
    def spawn_initial_neutrons(self, N):
        """
        depend on the geometry function and energy distribution
        """
        neutron_seq = []
        for i in range(N):
            neutron = Neutron(self.spawn_neutron(), watts_distribution(), unit_vector())
            neutron_seq.append(neutron)
        return neutron_seq



""" Define Physical Functions """
#
def unit_vector():
    """
    This is a random process
    """
    while True:
        sample = rnd.uniform(-1,1,size=3)
        if np.linalg.norm(sample)<=1:
            return sample/np.linalg.norm(sample)
        else:
            continue

def watts_distribution():
    a = 1
    b = 2
    K = 1 + (b/(8*a))
    L = 1/a*(K+(K**2-1)**(1/2))
    M = a*L - 1
    p1 = rnd.uniform(0,1)
    p2 = rnd.uniform(0,1)
    x = -np.log(p1)
    y = -np.log(p2)
    if (y-M*(x+1))**2 <= b*L*x:
        return 1000000*L*x
    else:
        return watts_distribution()

"""OOP functions"""

""" Define Main Functions """
def random_walk(seq):
    """
    Note we copy a array here, it may slow things
    """
    neutrions_seq = copy.deepcopy(seq)
    next_seq = np.array([])
    
    for neutron in neutrions_seq:
        
        while neutron.alive == True:

            medium = reactor.check_medium(neutron.position) #TODO let it depend on geometry
            
            neutron.move(medium)
            
            medium = reactor.check_medium(neutron.position)
            if medium == "out":
                
                neutron.alive = False
                continue
            elif neutron.collision(medium) == True:
                next_seq = np.hstack((next_seq, neutron.new_neutrons))
    
    return next_seq

""" Define Plots"""

def radius_distri(neutron_seq,title=time.perf_counter()):
    path1_dis = []
    for i in neutron_seq:
        path1_dis.append(np.linalg.norm(i.position))
    fig1, ax1 = plt.subplots(1, 1)
    ax1.hist(path1_dis, 50, density=True, alpha=0.75)
    title = time.perf_counter()
    fig1.suptitle(str(title), fontsize=16)
    plt.show()

""" Define cycles """


def cycle_0(n,number,initial_neutrions_0):
    """
    Here we copy the arrays, this may slow the process
    """
    x = [initial_neutrions_0]

    for i in range(n):
        if i == 0:
            sample = randomcopy(number,random_walk(initial_neutrions_0))
            x.append(sample)
        else:
            
            sample = randomcopy(number,random_walk(x[-1]))
            x.append(sample)
    return x

def cycle_1(n,number,initial_neutrions_0):
    x = [initial_neutrions_0]
    k = []
    for i in range(n):
        if i == 0:
            sample = random_walk(initial_neutrions_0)
            k.append(len(sample))
            x.append(sample)
        else:
            
            sample = random_walk(x[-1])
            k.append(len(sample))
            x.append(sample)
    return x,k


def randomcopy(n,neutron_seq):
    if len(neutron_seq) >= n:
        return neutron_seq[:n]
    if len(neutron_seq) <= n:
        while len(neutron_seq) < n:
            position = rnd.choice(neutron_seq,1)[0].position
            neutron = Neutron(position, watts_distribution(), unit_vector())
            neutron_seq = np.hstack((neutron_seq,neutron))
        return neutron_seq
            









#%%
"""Set up the mediums"""

U235 = Atom(u235)
U238 = Atom(u238)
O16 = Atom(o16)
U235.mass = openmc.data.atomic_mass('U235')
U238.mass = openmc.data.atomic_mass('U238')
O16.mass = openmc.data.atomic_mass('O16')

fuel = Medium(19.1, [[U235,0.05,1],[U238,0.95,1]])
richfuel = Medium(19.1, [[U235,0.8,1],[U238,0.2,1]])
realfuel_0 = Medium(10.97, [[U235,1,1],[O16,1,2]])
realfuel = Medium(10.97, [[U235,0.05,1],[U238,0.95,1],[O16,1,2]])
pureu235 = Medium(19.1, [[U235,1,1]])
pureu238 = Medium(19.1, [[U238,1,1]])


""" Set up the geometry"""

sphere = Geometry("sphere", r=0.68)
cylinder = Geometry("cylinder",r=3,z=8)
rectangle = Geometry("rectangle",x=1,y=3,z=3)


reactor = sphere
reactor.medium = realfuel




#%%
""" Set up the initial distribution"""

numberpercycle = 1000
numberofcycle = 30
numberof2cycle = 10
#%%
"""Start"""
start = time.perf_counter()
initial_neutrions = reactor.spawn_initial_neutrons(numberpercycle)


cycles = cycle_0(numberofcycle,numberpercycle,initial_neutrions)

end = time.perf_counter()
print("time: " + str(end - start))
""" Write the cycles into a small file"""
#%%
composition = "u235_5%_u238_95%_UO2"
filename = str(numberpercycle) + "n_" + str(numberofcycle)+ "cycles_" \
          + composition + reactor.shape +str(reactor.radius)+".pickle"
pickling_on = open(filename,"wb")
pickle.dump(cycles, pickling_on)
pickling_on.close()
#%%
""" Read the small file """

#pickle_off = open("cycles_0.83_pure.pickle","rb")
#cycles = pickle.load(pickle_off)






""" Examine the equilibrium """



# generation_0 = initial_neutrions
# generation_z = cycles[-1]


# radius_distri(generation_0)
# radius_distri(generation_z)

#%%
# calculate the average k
cycle2 = cycle_1(numberof2cycle,numberpercycle,cycles[-1])

skip = False
k_effs = []
for i in range(len(cycle2[1])):
    if skip == False:
        skip = True
    else:
        k_eff = cycle2[1][i]/cycle2[1][i-1]
        k_effs.append(k_eff)

k_array = np.array(k_effs)
print(reactor.radius)
print("number per cycle: "+str(numberpercycle))
print("number of cycle: "+str(numberofcycle))
print("k_avg: "+ str(np.average(k_array)))
print("std_k: " + str(np.std(k_array)))
end = time.perf_counter()
print("time: " + str(end - start))
#%%
start = time.perf_counter()
print(len(random_walk(cycles[-1])))
end = time.perf_counter()
print("time: " + str(end - start))

