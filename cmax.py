#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

## $\bar{C}_\text{max}(1 - \alpha, \mu)$

# Computing $\bar{C}_\text{max}(C, \mu)$ for optimum interval calculation, where $\mu$ is the number of expected events and $1 - \alpha$ is how frequently you reject the null hypothesis when it is true.

# The <b>single-event energy spectrum</b>, that is, the probability density function which tells us which energy depositions are likely to occur, is independent of the chosen WIMP model -- we always expect a simple exponential recoil spectrum.
# 
# The <b>number of dark matter events</b> detected does depend on the WIMP mass and cross-section. We know, however, that it must follow a Poisson distribution, which leaves the Poisson mean (which equals the expected number of events) as the only parameter left to estimate. From an upper limit on this mean, an upper limit curve in the dark matter mass – cross-section plane can be computed.
# 
# *	A <b>list_of_energies</b> list of reconstructed energy depositions of single events (from here on simply ‘energies’), either measured during some run of an actual detector, or generated using Monte Carlo.)
# *	An <b>interval</b> is an interval in energy space.
# *	The <b>size</b> of an interval is the fraction of energies expected in that interval. Clearly, this depends on which energy spectrum we assume, but is independent of the Poisson mean we are trying to constrain. By definition this is a number between 0 and 1.
# *	The <b>K-largest</b> interval of a run is the largest interval containing K events in that run. Recall our definition of size: a ‘large’ interval is one which is unusually empty in that run. Clearly k-largest intervals will terminate at (or technically, just before) an observed energy, or at one of the boundaries of our energy space. Again, which interval in a run is the k–largest, depends on our energy spectrum, but not on our Poisson mean.
# *	The <b>extremeness</b> of a K-largest interval is the probability of finding the K-largest interval in a run to be smaller. This clearly does depend on the Poisson mean: if we expect very few events, large gap sizes are more likely. Clearly extremeness is a number between 0 and 1; values close to 1 indicate unusually large intervals, that is, usually large (almost-)empty regions in the measured energies.
#  For example, if the extremeness of a k-largest interval in a run is 0.8, that means that 80% of runs have k-largest intervals which are smaller than the k-largest interval in this run.
# *	The <b>optimum interval statistic</b> of a run is extremity of the most extreme k-largest interval in a run.
# *	The <b>extremeness</b> of the optimum interval statistic is the probability of finding a lower optimum interval statistic, that is, of finding the optimum interval in a run to be less extreme. 
# 
# The <b>max gap method</b> rejects a theory (places a mean outside the upper limit) based on a run if the 0-largest interval (the largest gap) is too extreme. 
# 
# The <b>optimum interval</b> method rejects a theory based on a run if the optimum interval statistic is too large.
# 
# * The <b>energy cumulant</b> $\epsilon(E)$ is the fraction of energies expected below the energy $E$. Whatever the (1-normalized) energy distribution $dN/dE$, $dN/d\epsilon$ is uniform[0,1], where $0$ and $1$ correspond to the boundaries of our experimental range.
# 
# 

# In[16]:
from __future__ import print_function
import functools
from scipy.optimize import brenth
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import gzip
import sys


# In[17]:

def kLargestIntervals(list_of_energies, spectrumCDF = lambda x: x):
    """
    Returns a list of the sizes of the K-largest intervals in that run according to the energy spectrum (given as a CDF).
    That is, kLargestIntervals(...)[i] is the size of the largest interval containing i events, where ‘largest’ is defined above.
    
    * Transform energies to energy cumulants
    * Add events at 0 and 1
    * Foreach k, compute interval sizes, take max
    """
    answer = {}
    
    list_of_energies.sort()
    
    energy_cumulants = spectrumCDF(list_of_energies)
        
    for interval_size in range(len(energy_cumulants)):
        if (1 + interval_size) >= len(energy_cumulants):
            continue
            
        temp_data = energy_cumulants.copy()
        gap_sizes = temp_data[(1+interval_size):] - temp_data[0:-1*(1 + interval_size)] 

        answer[interval_size] = np.max(gap_sizes)

    return answer

assert kLargestIntervals(np.array([0.0, 0.1, 0.2, 0.84, 0.85]))[0] == (0.84 - 0.2)  # test 0
assert kLargestIntervals(np.array([0.0, 0.1, 0.2, 0.84, 0.85]))[2] == (0.84 - 0.0)  # test 2
assert kLargestIntervals(np.array([0.85, 0.0, 0.1, 0.84, 0.2]))[2] == (0.84 - 0.0)  # test unsorted


# In[18]:

def extremenessOfInterval(x, k, mu):
    """
    Returns the extremeness of a k-largest interval of size, if the poisson mean is mu.
    
    (Number of itvSizes[mu][k] smaller than size) / mcTrials[mu]
    
    x - also size in above comment
    k - gap (rename k)
    """
    # [0] is because where returns list, where [0] is answer
    if k not in itvSizes[mu]:
        return 0
    
    return np.where(itvSizes[mu][k] < x)[0].size / mcTrials[mu]


# In[19]:

def optimumItvStatistic(list_of_energies, mu, spectrumCDF = lambda x: x):
    """
    Returns the optimum interval statistic of the run.
    
    Max of extremenssOfInterval's
    """
    return np.max([extremenessOfInterval(x, k, mu) for k, x in kLargestIntervals(list_of_energies, spectrumCDF).items()])


# In[20]:

def extremenessOfOptItvStat(stat, mu):
    """
    Returns the extremeness of the optimum interval statistic stat, given mu
    
    (Number of optItvs[mu] smaller than stat) / mcTrials[mu]
    """
    return np.where(optItvs[mu] < stat)[0].size / mcTrials[mu]


# In[26]:

def optItvUpperLimit(list_of_energies, c, spectrumCDF = lambda x: x,
                     n = 10000):
    """
    Returns the c- confidence upper limit on mu using optimum interval
    
    For which mu is extremenessOfOptItvStat( optimumItvStatistic(run), mu ) = c

    Does divide and conquer
    
    c - e.g., 0.9
    """
    def f(mu, list_of_energies, c, spectrumCDF, n):
        """Can be used for optimizations too"""
        generate_table(mu, n)
        x = optimumItvStatistic(list_of_energies, mu, spectrumCDF)
        prob = extremenessOfOptItvStat(x, mu)
        return prob - c

    mu = 0

    test_mus = np.arange(10,
                         (2*list_of_energies.size))
    
    def has_min(values):
        x = f(values[0], list_of_energies, c, spectrumCDF, n)
        y = f(values[-1], list_of_energies, c, spectrumCDF, n)
        
        if x < 0 and y > 0:
            print('[%d, %d]: continue\t%f %f' % (values[0], values[-1],
                                                 x, y))
            return True
        else:
            print('[%d, %d]:  deadend\t%f %f' % (values[0], values[-1],
                                                 x, y))
            return False
    
    def split_search(values):
        if values.size == 1:
            return values[0]
        a, b = np.array_split(values, 2)

        if has_min(a):
            return split_search(a)
        elif has_min(b):
            return split_search(b)
        elif has_min(np.array([a[-1], b[0]])):
            #print(values, a, b)
            return b[0]
        else:
            print(values)
            raise RuntimeError('no minimum?')

    return split_search(test_mus)

    #for mu in np.arange(10, 2 * list_of_energies.size):
    #    if f(mu, list_of_energies, c, spectrumCDF, n) > 0:
    #        return mu


    


# In[22]:

def generate_trial_experiment(mu, n):
    trials = []

    for index in range(n):
        this_mu = np.random.poisson(mu)
        
        rand_numbers = np.random.random(size=this_mu)
        rand_numbers = np.append(rand_numbers, [0.0, 1.0])
        rand_numbers.sort()
        trials.append(rand_numbers)

    return trials


### Monte Carlo for populating itvSizes[$\mu$][$k$] and optItvs[$\mu$]

# In[36]:

def get_filename(mu = None):
    if mu:
        return 'saved_intervals_%04d.p.gz' % mu
    else:
        return 'saved_intervals.p.gz'

def load_table_from_disk(mu = None):
    global itvSizes
    global optItvs
    global mcTrials
    
    if os.path.exists(get_filename(mu)):
        with gzip.open(get_filename(), 'rb') as f:
            itvSizes = pickle.load(f)
            optItvs = pickle.load(f)
            mcTrials = pickle.load(f)
    
def write_table_to_disk(mu = None):
    with gzip.open(get_filename(mu), 'wb') as f:
        pickle.dump(itvSizes, f)
        pickle.dump(optItvs, f)
        pickle.dump(mcTrials, f)


itvSizes = {}
optItvs = {}
mcTrials = {}
#load_table_from_disk()    
    
def generate_table(mu, n):
    """    #Generate trial runs"""    
    if mu in mcTrials and mcTrials[mu] >= n:
        return

    print("Generating", mu)

    mcTrials[mu] = n
    trials = generate_trial_experiment(mu, mcTrials[mu])

    itvSizes[mu] = {}
    optItvs[mu] = []

    for trial in trials:
        intermediate_result = kLargestIntervals(trial)
        
        for k, v in intermediate_result.items():
            if k not in itvSizes[mu]:
                itvSizes[mu][k] = []

            itvSizes[mu][k].append(v)
    
    # Numpy-ize it
    for k, array in itvSizes[mu].items():
        itvSizes[mu][k] = np.array(array)
    
    for trial in trials:
        optItvs[mu].append(optimumItvStatistic(trial, mu))
        
    # Numpy-ize it
    optItvs[mu] = np.array(optItvs[mu])
    
    


def cache_values(my_max=200, n=100):
    for i in range(3, my_max):
        generate_table(i, n)
    write_table_to_disk()

if __name__ == "__main__":
    mu = 10
    n = 1000

    if len(sys.argv) > 1:
        mu = int(sys.argv[1])
    elif len(sys.argv) > 2:
        n = int(sys.argv[2])
    
    print('mu', mu, 'n', n)
    generate_table(mu, n)
    write_table_to_disk(mu)
