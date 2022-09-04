### GIVES YOU POTENTIAL CANDIDATE BINARY SYSTEMS BASED ON CONSTRAINS FROM CORSE GRAINED DATA AND PLOTS INTERESTING QUANTATIES (SEP,M,Lum,..)

import numpy as np
import pickle as pickle
from numpy.lib.function_base import append
import yt as yt
import matplotlib.pyplot as plt
import sys

from yt.utilities.physical_ratios import newton_cgs

units = {'length_unit': yt.YTQuantity(4.0, 'pc'),
         'velocity_unit': yt.YTQuantity(0.18, 'km/s'),
         'time_unit': yt.YTQuantity(685706129102738.9, 's'),
         'mass_unit': yt.YTQuantity(3000.0, 'Msun'),
         'density_unit': yt.YTQuantity(46.875, 'Msun/pc**3')}

pickle_file = "/lustre/astro/troels/IMF_512/stars_red_512.pkl"

file_open = open(pickle_file, 'rb')
global_data = pickle.load(file_open, encoding='latin1')
file_open.close()

# # # CONSTANTS ON WHICH WE BASE OUR SEARCH
plotting = True 
final_time = 0.3            # 200 kyr after the formation
neighbour_constraint = 10000  # records sinks that have a maximum separation of 3000 au

# Mass of sink 1 at final_time after time of creation
lower_m = yt.YTQuantity(0.35, 'Msun')
higher_m = yt.YTQuantity(6, 'Msun')

# how much younger we allow sink2 to be
sink2_formation_wait_time = yt.YTQuantity(0.03, 'Myr')

# Upper boundary for system mass during the time t_creation(primary) + final_time
max_sysMass = yt.YTQuantity(4, 'Msun')


# X positions of sink 0 at all times (nt= number of timesteps)
nt = len(global_data['X'][:, 0])

# X position of all sinks at time = 0
nsink = len(global_data['X'][0, :])



# TROELS SAYS:
def periodic_distance(p1, p2):
    if np.any(p1.shape != p2.shape):
        print(p1.shape,p2.shape)
        sys.exit("periodic distance called with un-equal shapes. Stopping")
    if np.any(p1<0) or np.any(p1>1) or np.any(p2<0) or np.any(p2>1):
        print(p1.min(),p1.max())
        print(p2.min(),p2.max())
        sys.exit('Positions out of bounds!')

    pp = np.minimum(np.minimum(abs(p1-p2), abs(p1-p2+1)), abs(p1-p2-1))
    return np.sqrt((pp**2).sum(axis=0))


def position(sink, t):
    return np.array([global_data['X'][t, sink], global_data['Y'][t, sink], global_data['Z'][t, sink]])

def system_mass(s1, s2, t):
    return np.array(global_data['M'][t, s1] + global_data['M'][t, s2])*units['mass_unit'].in_units('Msun')

def sink_mass(s1, t):
    return np.array(global_data['M'][t, s1])*units['mass_unit'].in_units('Msun')

def creation_time(sink):
    return global_data['T'][-1, sink]*units['time_unit'].in_units('Myr')


def last_sink(f_time):
    # Gives the sinks that have been evolved for at least  f_time
    last_time = global_data['TIME'][-1, -1]*units['time_unit'].in_units('Myr')
    return np.where(global_data['T'][-1, :]*units['time_unit'].in_units('Myr') + yt.YTQuantity(f_time, 'Myr') < last_time)[0][-1]


def system_mass_at_3_times(s1, s2):
    time_array1 = global_data['TIME'][:, s1]*units['time_unit'].in_units('Myr')
    time_array2 = global_data['TIME'][:, s2]*units['time_unit'].in_units('Myr')
    t1 = yt.YTQuantity(final_time/4, 'Myr')
    t2 = yt.YTQuantity(final_time/2, 'Myr')
    t3 = yt.YTQuantity(final_time, 'Myr')
    return [(global_data['M'][np.where(time_array1 > creation_time(s1) + t1)[0][0], s1] + global_data['M'][np.where(time_array2 > creation_time(s1) + t1)[0][0], s2])*units['mass_unit'].in_units('Msun'),
            (global_data['M'][np.where(time_array1 > creation_time(s1) + t2)[0][0], s1] + global_data['M']
             [np.where(time_array2 > creation_time(s1) + t2)[0][0], s2])*units['mass_unit'].in_units('Msun'),
            (global_data['M'][np.where(time_array1 > creation_time(s1) + t3)[0][0], s1] + global_data['M'][np.where(time_array2 > creation_time(s1) + t3)[0][0], s2])*units['mass_unit'].in_units('Msun')]



def time_inside_final_time(sink,final_time):
    # Time indexes of the period of final_time after first sink formation
    t_creation1 = creation_time(sink)
    times1 = global_data['TIME'][:, sink]*units['time_unit'].in_units('Myr')
    t_creation200 = t_creation1 + yt.YTQuantity(final_time, 'Myr')
    return np.where((times1 > t_creation1) & (times1 < t_creation200))[0]


def accretion_rate(sink,t):
    dm = global_data['dm'][t,sink]*units['mass_unit'].in_units('Msun')
    dt = (global_data['time'][t,sink] - global_data['tflush'][t,sink])*units['time_unit'].in_units('yr')
    return dm/dt

def luminosity(sink,t):
    G = yt.YTQuantity(newton_cgs, "cm**3/g/s**2")
    R_sun = yt.YTQuantity(1, "Rsun")
    return (0.5*G*sink_mass(sink,t)*accretion_rate(sink,t)/(4*R_sun)).in_units('Lsun')

# TROELS SAYS:
def COM_direction(x,m):
    x0 = x[0] # get reference points for each timestep
    nsink = (x.shape)[0]
    ntime = (x.shape)[1]
    xx = np.zeros((nsink,ntime)) # array holding recentered coordinates
    dd = np.zeros((3,nsink,ntime)) # array for possible shifted coordinates
    j = range(ntime)
    for sid in range(nsink):
      dd = np.array([x[sid,:] - x0, x[sid,:] - x0 + 1, x[sid,:] - x0 - 1])
      i = np.argmin(abs(dd),axis=0)
      xx[sid] = (dd[i,j] + x0)* m[sid] # sink mass weight
    #print(xx.shape)
    COM = np.sum(xx,axis=0) / m.sum(axis=0)  
    #print(COM.shape)

    COM[COM < 0] += 1
    COM[COM >= 1.] -= 1

    return COM

def com_position(p,m):
    ntime = (p.shape)[2]
    com = np.zeros((3,ntime))
    # position is knowing all directions
    for i in range(3):
        com[i] = COM_direction(p[:,i,:],m)  # given coordinate of both sinks at all times (2,ntime)
    return com



### ------ SEARCHING FOR CANDIDATES ----- (AKA MAIN 1)
max_sink = last_sink(final_time)

all_closest_aproaches = [] # closest aproaches of all sink2 for all sink1 of 1 - 3 Msun inside 200 kyr
candidates = []

# Sink1 is the primary
for sink1 in range(max_sink):
    sink1_times = global_data['TIME'][:, sink1]*units['time_unit'].in_units('Myr')
    
    t_creation1 = creation_time(sink1)
    t_creation100 = t_creation1 + yt.YTQuantity(final_time/2, 'Myr')
    t_creation200 = t_creation1 + yt.YTQuantity(final_time, 'Myr')

    
    time_indexes_inside200 = time_inside_final_time(sink1,final_time) # Time indexes of the period of final_time after first sink formation
   
    mass1_at_200ky = global_data['M'][np.where(sink1_times > t_creation200)[0][0], sink1]*units['mass_unit'].in_units('Msun') # INDEXING: global_data['M'][np.where(times > t_creation1),sink] gives an [[n1,n2...]] so the first [0] is to acces the right array and the second [0] is to find the timestep that is closest to the targetet t=t_creation + 0.1 Myr

    neighbours = []  # sinks inside a neighbourhood

    # # # Test if the mass of sink1 is between 1 and 3 msun at 200 kyr
    if mass1_at_200ky < higher_m and mass1_at_200ky > lower_m:
        for sink2 in range(max_sink):
            if sink1 != sink2:  # there has to be a smarter way of doing this...
                t_creation2 = creation_time(sink2)
                closest_aproach = np.min(periodic_distance(position(sink1,time_indexes_inside200), position(sink2,time_indexes_inside200) )*units['length_unit'].in_units('au'))
                all_closest_aproaches.append((sink1, sink2, closest_aproach))
                # if closest aproach is inside the neighbourhood coinstraint --> IT'S A NEIGHBOUR
                if closest_aproach < neighbour_constraint:
                    neighbours.append(sink2)

        # # # Consolidate the list of neighbors of a given sink and test that it is <= 2 (or 3)
        if len(neighbours) <= 3:
            # Check for those neighbors the time difference in creation to the first one. We do not want to wait more than ≈25 kyr for the secondary to form.
            timed_neighbours = [k for k in neighbours if abs(creation_time(k) - creation_time(sink1)) < sink2_formation_wait_time]
            for neigh in timed_neighbours:
                # # # Test that the system mass is aways < 4 msun inside 200 kyr
                if np.all(system_mass(sink1, neigh, time_indexes_inside200) < max_sysMass):
                    candidates.append((sink1, neigh))


print(candidates) ### FOUND CANDIDATES 

# TO DO: find multiple systems (>2)

# putting whipped cream and a cherry on top 
caps = np.array(all_closest_aproaches, dtype=[('sink1', 'i4'), ('sink2', 'i4'), ('ca', 'f8')])

list_of_records = []
for sinks in candidates:
    sorted_closest_approaches = sorted(caps[np.where(caps['sink1'] == sinks[0])], key=lambda tup: tup[2])[:3]  # first 3 for a given candidate primary
    record_dict = {'Sink1': sinks[0], 'sink2': sinks[1], 'time_diff':  creation_time(sinks[1]) - creation_time(sinks[0]), 'closest_approach': sorted_closest_approaches, '3Masses': system_mass_at_3_times(sinks[0], sinks[1])}
    list_of_records.append(record_dict)


for n in range(len(list_of_records)):
    print(list_of_records[n], "\n")


### ANALYSIS AND PLOTTING INTERESTING DATA (AKA MAIN 2)
if plotting == True :

    for count, candidate in enumerate(candidates):
        
        time_indexes_inside_200kyr = time_inside_final_time(candidate[0],final_time)
        time = 1000*(global_data['TIME'][time_indexes_inside_200kyr,candidate[0]]*units['time_unit'].in_units('Myr') - creation_time(candidate[0]))

        sink_masses = np.array([sink_mass(k,time_inside_final_time(candidate[0],final_time)) for k in candidate])
        positions =   np.array([ position(k,time_inside_final_time(candidate[0],final_time)) for k in candidate])
        position_com = com_position(positions,sink_masses)
        nsinks = np.shape(positions)[0]

        # TU SI SJEBO NES ----
        com_d = np.array([periodic_distance(positions[k],position_com)*units['length_unit'].in_units('au') for k in range(nsinks)])
        #print(com_d)
        #print(np.min(com_d))
        #print(np.max(com_d))

        fig, ax = plt.subplots(4,1,sharex='all',figsize=(16, 14))
        
        ### DISTANCE FROM CENTER OF MASS
        ax[0].plot(time, com_d[0], '-', label=f'Sink {candidate[0]:2d} ')  # COM DISTANCE OF SYSTEM
        ii = np.where(sink_masses[1] > 0)
        #ax[0].plot(time[ii], com_d[1][ii], '-', label=f'Sink {candidate[1]:2d} ') 
        #print('com0:',com_d[0],com_d[0].shape,'com1',com_d[1],com_d[1].shape)
        list_of_aproaches = list_of_records[count]['closest_approach']

                                                            
        for approach in range(len(list_of_aproaches)-1):  # COM DISTANCE OF CLOSEST APROACHES
            close_sink = list_of_aproaches[approach][1]
            close_com_d = periodic_distance(position(close_sink,time_indexes_inside_200kyr),position_com)*units['length_unit'].in_units('au')
            ii = np.where(sink_mass(close_sink,time_indexes_inside_200kyr)>0)
            ax[0].plot(time[ii], close_com_d[ii], '--', label=f'Sink {close_sink:2d} ')

        ax[0].set(xlim=(time[0], time[-1]),
                #ylim=(4*10**5, 6*10**6),
                #title=f'Distance from the center of mass of sinks {candidate[0]:2d} and {candidate[1]:2d}',
                xlabel='time [kyr]',
                ylabel=' Distance from center of mass [AU] ')
        ax[0].set_yscale('log')
        ax[0].set_ylim(1e-1, 1e5)
        ax[0].legend(loc='upper right')

        ### MASSS
        s_mass = system_mass(candidate[0],candidate[1],time_indexes_inside_200kyr)
        m1 = sink_mass(candidate[0],time_indexes_inside_200kyr)
        m2 = sink_mass(candidate[1],time_indexes_inside_200kyr)
        
        ax[1].plot(time,m1,label=f'Mass of sink {candidate[0]}')
        ax[1].plot(time,m2,label=f'Mass of sink {candidate[1]}')
        ax[1].plot(time,s_mass,label='System mass')

        ax[1].set(xlim=(time[0], time[-1]),
                #ylim=(4*10**5, 6*10**6),
                #title=f'Mass of {candidate[0]:2d} and {candidate[1]:2d}',
                xlabel='time [kyr]',
                ylabel=r' Mass [$M_\odot $] ')
        #ax[1].set_yscale('log')
        #ax0.set_ylim(4*10**5, 6*10**6)
        ax[1].legend(loc='upper right')

        ### ACCRETION RATE
        ax[2].plot(time,accretion_rate(candidate[0],time_indexes_inside_200kyr),label=f'Accretion rate of sink {candidate[0]:2d}')
        ax[2].plot(time,accretion_rate(candidate[1],time_indexes_inside_200kyr),label=f'Accretion rate of sink {candidate[1]:2d}')
        ax[2].plot(time,accretion_rate(candidate[0],time_indexes_inside_200kyr) + accretion_rate(candidate[1],time_indexes_inside_200kyr),label=f'Accretion rate of sink {candidate[0]:2d} + {candidate[1]:2d}')

        ax[2].set(xlim=(time[0], time[-1]),
                #ylim=(4*10**5, 6*10**6),
                #title=f'Mass accretion rate of sinks {candidate[0]:2d} and {candidate[1]:2d}',
                xlabel='time [kyr]',
                ylabel=r' Accretion rate  $\left [ \frac{\mathrm{M}_\odot}{\mathrm{yr}} \right ]$ ')
        ax[2].set_yscale('log')
        #ax0.set_ylim(4*10**5, 6*10**6)
        ax[2].legend(loc='lower left')

        ### LUMINOSITY
        ax[3].plot(time,luminosity(candidate[0],time_indexes_inside_200kyr),label=f'Luminosity of sink {candidate[0]:2d}')
        ax[3].plot(time,luminosity(candidate[1],time_indexes_inside_200kyr),label=f'Luminosity of sink {candidate[1]:2d}')
        ax[3].plot(time,luminosity(candidate[0],time_indexes_inside_200kyr) + luminosity(candidate[1],time_indexes_inside_200kyr),label=f'Luminosity of sink {candidate[0]:2d} + {candidate[1]:2d}')

        ax[3].set(xlim=(time[0], time[-1]),
                #ylim=(4*10**5, 6*10**6),
                #title=f'Luminosity of {candidate[0]:2d} and {candidate[1]:2d}',
                xlabel='time [kyr]',
                ylabel=r'Luminosity $ \left [ \mathrm{L}_\odot \right ] $')
        ax[3].set_yscale('log')
        #ax0.set_ylim(4*10**5, 6*10**6)
        ax[3].legend(loc='lower left')


        plt.savefig(f'/lustre/astro/vitot/results/zoom_in_candidates/zoom_in_candidates_s1_{int(candidate[0]):2d}_s2_{int(candidate[1]):2d}_nc_{int(neighbour_constraint):4d}_ft_{int(100*final_time):3d}kyr.png',bbox_inches='tight', dpi=250)




# Figure out closest approach of nearest, second nearest and third nearest sink particle during 200 kyr
# printout the above characteristics: sink numbers, difference in formation, closest approach of 1st, 2nd, and third sink. Mass of system at 50 kyr, 100 kyr and 200 kyr after primary sink formation.

#sorted_closest_approaches = np.array(sorted_closest_approaches, dtype=[('sink1', 'i4'), ('sink2', 'i4'), ('ca', 'f8')])
#print('3 Closest approaches of sink 2:',sorted_closest_approaches[np.where(sorted_closest_approaches['sink1'] == 59)[0]])
#print(all_closest_aproaches[np.where( all_closest_aproaches[:][0]==89 )])

# sinkic = candidates[0][0]
# print(sorted(caps[np.where(caps['sink1'] == sinkic)],key=lambda tup: tup[2] )[:3]) # 3 closest aproaches for a given sink


'''   
            
            # Sink2 is created after Sink1 AND we do not want to wait more than ≈25 kyr = 0.025 Myr for the secondary to form AND we dont want the system mass to be larger than 4 Msun inside 200 kyr
            if t_creation2 > t_creation1 and t_creation2 < t_creation1 + yt.YTQuantity(0.025,'Myr') and np.all( system_mass(sink1,sink2, time_indexes_inside200) < yt.YTQuantity(4, 'Msun')):
                                          # sink2 creation time comstraint                                              system mass constraint                                           
                
                #closest_aproach = np.min(separation( sink1,sink2,time_indexes_inside200 )) 

                  
                    #sink_pairs.append((sink1,sink2))


                   time_diff = t_creation1 - t_creation2
                    
                    times2 = global_data['TIME'][:,sink2]*units['time_unit'].in_units('Myr') 
                    sysMass = system_mass_at_3_times(sink1,sink2,times1,times2,t_creation50,t_creation100,t_creation200)
                
                    sink1_closest_approaches_masses_timediff.append( (sink2, np.min(distances),sysMass,time_diff) ) 
'''


'''
        sorted_closest_approaches = sorted(sink1_closest_approaches_masses_timediff, key=lambda tup: tup[1]) # sorting based on the second element of a tuple (closest approaches)
        sink_best = sorted_closest_approaches[0][0]
        
        closest3 = sorted_closest_approaches[:3][0:1]
        time_diff_best = sorted_closest_approaches[0][3]
        systemMass = sorted_closest_approaches[0][2]


        record_dict={'Sink1': sink1, 'sink2': sink_best, 'time_diff' : time_diff_best, 'closest_approach' : closest3, '3Masses' : systemMass } #append this for all sinks
        print(record_dict)
        list_of_records.append(record_dict)   

print(len(list_of_records))'''
# TO DO: 1) record distance (sink1,sink2) for all timsteps during 200 kyr --> min is the closest aproach
#        2) organise data to figure out closest approach of nearest, second nearest and third nearest sink particle during 200 kyr


'''

    # Check if the mass of a sink1 IS between 1 and 3 msun FOR ALL PERIOD OF tc1+200 kyr
    t_creation1 = global_data['T'][-1,sink1]*units['time_unit'].in_units('Myr')
    t_creation100 = global_data['T'][-1,sink1]*units['time_unit'].in_units('Myr') + yt.YTQuantity(0.1, 'Myr')
    t_creation200 = global_data['T'][-1,sink1]*units['time_unit'].in_units('Myr') + yt.YTQuantity(0.2, 'Myr')
    times = global_data['TIME'][:,sink1]*units['time_unit'].in_units('Myr') 
    mass = global_data['M'][np.where(times > t_creation100),sink1][0][0]*units['mass_unit'].in_units('Msun') 
    lower_m = yt.YTQuantity(1,'Msun')
    higher_m = yt.YTQuantity(3,'Msun')
    if np.all(mass < higher_m and mass > lower_m) :


com_d = com_distance_tupleTROELS(candidate, time_indexes_inside_200kyr)
comas = center_of_mass(candidate[0], candidate[1], time_indexes_inside_200kyr)
'''
'''
OLD FUNCTIONS:

# dont use this

def separation(s1, s2, t):
    return np.linalg.norm(position(s1, t) - position(s2, t), axis=0)*units['length_unit'].in_units('au')

def center_of_mass(s1, s2, t):
    total_mass = system_mass(s1, s2, t)
    m1 = global_data['M'][t, s1]*units['mass_unit'].in_units('Msun')
    m2 = global_data['M'][t, s2]*units['mass_unit'].in_units('Msun')
    r1 = position(s1, t)
    r2 = position(s2, t)
    #dis = periodic_distance(r1,r2)
    return (m1*r1 + m2*r2)/total_mass


def com_distance_tuple(candidate, t):
    # [x,] is the dimension (x,y,z) [,x] is the timestep
    com = center_of_mass(candidate[0], candidate[1], t)[0, :]

    com_distance1 = np.linalg.norm(
        position(candidate[0], t) - com, axis=0)*units['length_unit'].in_units('au')
    com_distance2 = np.linalg.norm(
        position(candidate[1], t) - com, axis=0)*units['length_unit'].in_units('au')
    return (com_distance1, com_distance2)


def com_distance_tupleTROELS(candidate, t):
    com = center_of_mass(candidate[0], candidate[1], t)
    com_distance1 = periodic_distance(position(candidate[0],t), com)*units['length_unit'].in_units('au')
    com_distance2 = periodic_distance(candidate[1], com)*units['length_unit'].in_units('au')
    return (com_distance1, com_distance2)


def sink_point_distance(s, t, point):
    return np.linalg.norm(position(s, t) - point, axis=0)*units['length_unit'].in_units('au')

def sink_point_distanceTROELS(s, t, point):
    sink_position = position(s, t)
    return periodic_distance(sink_position, point)
'''