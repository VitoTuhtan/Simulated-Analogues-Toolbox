#!/bin/python
#begin_time = datetime.datetime.now()

'''Post-procesing high resolution simulation data using tools from yt module. 
Callculates angular momentum, velocities in cylindrical coordinates in order to find disk size around sink particles.
If a sink particle is a part of a binary system it also callculates separation and Roche radius.'''


from mpi4py.MPI import COMM_WORLD as CW
import time 
import datetime
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pyramses import rsink
import glob
import yt 
#yt.set_log_level(40)
import argparse
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-s1", "--sink1", help="Sink number of the primary for which you want the analysis to be done for. First created sink is labeld with 0")
parser.add_argument("-s2", "--sink2", help="Sink number of the secondary")
parser.add_argument("-d", "--data", help="Sink data directory")
parser.add_argument("-save", "--save", help="Where to save the output")
parser.add_argument("-verbose", "--verbose", help="PRINT MORE OUTPUT",default=False)
args = parser.parse_args()


data_directory = args.data
save_dir =args.save
debug_verbose = args.verbose
sink_tag = int(args.sink1)
if args.sink2 == 'none' or args.sink2 == 'None' or args.sink2 == 'n':
    secondary_sink_tag = None
else:
    secondary_sink_tag = int(args.sink2)

rank = CW.Get_rank()
size = CW.Get_size()

# SCRIPT ARGUMENTS:

h_unit = 'au**2/yr' 
v_unit = 'au /yr'  
m_unit = 'Msun'      
first_one = 2
last_one = -1
freq=1

nshells = 30
rmax = 500 # in au
e=0.25 #height of the disk: hs=e*rs

#first_one = 1724-145  #if you don't want all of the outputs to be calculated
#first_one = 300-188 # 90
#first_one = 600 # 90

#sink_tag = 240
#secondary_sink_tag = 241

#data_directory = '/lustre/astro/vitot/L20_sink_240/data'
#data_directory = '/lustre/astro/troels/IMF_512_cores/sink_92/data'
#data_directory = '/lustre/astro/troels/IMF_512_cores/sink_49/data' #1724 last one!
#data_directory ='/lustre/astro/troels/IMF_512_cores/sink_165/rerun_l18/data'



############### SETTING UP THE SCRIPT: 
units = {'length_unit': yt.YTQuantity(4.0, 'pc'),
         'velocity_unit': yt.YTQuantity(0.18, 'km/s'),
         'time_unit': yt.YTQuantity(685706129102738.9, 's'),
         'mass_unit': yt.YTQuantity(3000.0, 'Msun'),
         'density_unit': yt.YTQuantity(46.875, 'Msun/pc**3')}

def _Density(field,data):
  #Overwrites density field 
  density_unit = (data.ds.mass_unit/data.ds.length_unit**3).in_cgs()
  density = data[('ramses', 'Density')].value*density_unit
  density_arr = yt.YTArray(density, 'g/cm**3')
  del density
  del density_unit
  return density_arr
yt.add_field('Density', function=_Density, units=r'g/cm**3')

def shape_dr_dv(yt_shape, com, v_com): 
    ''' Find the coordinates and velocities in a given frame of refference (center of mass) '''
    #dens = yt_shape['Density'].value*units['density_unit'].in_units('g/cm**3')
    dx = yt_shape['x'] - com[0]
    dy = yt_shape['y'] - com[1]
    dz = yt_shape['z'] - com[2]
    
    d_ux = yt_shape['x-velocity'] - v_com[0]
    d_uy = yt_shape['y-velocity'] - v_com[1]
    d_uz = yt_shape['z-velocity'] - v_com[2]

    dr = yt.YTArray([dx, dy, dz])
    dv = yt.YTArray([d_ux, d_uy, d_uz]) #rows are xyz, accesed by [x,y,z]
    return dr, dv

def shape_angular_momentum(yt_shape,sink_position,sink_velocity): # vito implementation.
    '''Calculate the angular momentum of the gass inside a given shape'''
    dens = yt_shape['Density'].value*units['density_unit']
    dr , dv = shape_dr_dv(yt_shape,sink_position,sink_velocity ) 
    mass = (dens*yt_shape['cell_volume']).sum()
    r_x_v = yt.ucross(dr,dv,axis=0 )
    cell_L = (dens*yt_shape['cell_volume']*r_x_v/mass).in_units(h_unit)
 
    L_gass = yt.YTArray( [cell_L[0].sum(),cell_L[1].sum(),cell_L[2].sum()] ) #IF THIS DOESNT WORK I AM THE POPE
    #return shape_angular_momentum(sphere,center,bulk_velocity) , mass.in_units(m_unit) 
    #print('Lgass:',L_gass)
    return L_gass


def shape_velocities(yt_shape,sink_position,sink_velocity,spin_axis,verbose = debug_verbose):
    #dens = yt_shape['Density'].value*units['density_unit'].in_units('g/cm**3')
    dr , dv = shape_dr_dv(yt_shape,sink_position,sink_velocity)
    
    e_r = dr/yt.YTArray(np.linalg.norm(dr,axis = 0),str(dr.units)) ### CHECK THIS AGAIN - checked.
    #print(np.linalg.norm(e_r,axis=1))
    ''' PRODUCTS '''
    time_steps = len(e_r[0,:])
    angular_vectors = yt.YTArray([ yt.ucross(spin_axis,e_r[:,k]) for k in range(time_steps)]).T
    angular_velocities = yt.YTArray([dv[:,k].dot(angular_vectors[:,k]) for k in range(time_steps)])   # np.multiply gives a Haddamar product (c_ij=a_ij*b_ij) -> sum of the collumns is the dot products of the vectors in the collumns ( 3 dim for example positions)
    
    radial_vectors = yt.YTArray([ yt.ucross(spin_axis,angular_vectors[:,k]) for k in range(time_steps)]).T
    radial_velocities = yt.YTArray([dv[:,k].dot(radial_vectors[:,k]) for k in range(time_steps)]) 
    z_velocites = yt.YTArray([dv[:,k].dot(spin_axis) for k in range(time_steps)])  #z direction is defined by s

    if verbose == True:
        print('dr',dr,np.shape(dr))
        print('Spin',spin_axis)
        print('z_velocities',z_velocites,np.shape(z_velocites))
        print('dv ',dv,len(dv),np.shape(dv))
        print('angular_vectors ',angular_vectors,len(angular_vectors),np.shape(angular_vectors))
        print('angular_velocities ',angular_velocities,len(angular_velocities),np.shape(angular_velocities))
        print('radial_velocities: ', radial_velocities, np.shape(radial_velocities))

    return angular_velocities , radial_velocities, z_velocites 

def substractable_values(yt_shape,sink_position,sink_velocity, spin_axis):
    dens = yt_shape['Density'].value*units['density_unit'].in_units('g/cm**3')
    #l_shape = shape_angular_momentum(yt_shape)
    angular_v , radial_v, z_v = shape_velocities(yt_shape,sink_position,sink_velocity,spin_axis)
    
    p_phi_shape = (dens*yt_shape['cell_volume']*angular_v).sum()
    
    p_r_shape = (dens*yt_shape['cell_volume']*radial_v).sum()
    p_z_shape = (dens*yt_shape['cell_volume']*z_v).sum()
    Mshape = (dens*yt_shape['cell_volume']).sum()

    p_phi_2shape = (dens*yt_shape['cell_volume']*angular_v**2).sum()
    p_r_2shape = (dens*yt_shape['cell_volume']*radial_v**2).sum()
    p_z_2shape = (dens*yt_shape['cell_volume']*z_v**2).sum()
    return p_phi_shape, p_r_shape, p_z_shape, p_phi_2shape, p_r_2shape,p_z_2shape, Mshape 

def m_sphere(centre,radius):
    sphere = ds.sphere(centre, (radius, "au"))
    dens = sphere['Density'].value*units['density_unit'].in_units('g/cm**3')
    return (dens*sphere['cell_volume']).sum()

def recursive_gass_spin_vector_refinment(sink_position,sink_velocity,disk_size=150,refinment_levels=7):
    # GASS SPIN VECTOR REFINMENT - angular momentum of the gass inside a big sphere gives us an estimate of the normal vector for the disk ->Ldisk1 gives a better estimate -> use Ldisk2 or 3
    # STARTING DISK SIZE?? 
    sphere = ds.sphere(sink_position, (1000, "au"))
    l_first = shape_angular_momentum(sphere,sink_position,sink_velocity)
    
    refined_s = l_first/yt.YTArray(np.linalg.norm(l_first),h_unit)  ### THIS
    print('S SPHERE:',refined_s)

    for n in range(refinment_levels):
        
        trial_disk = ds.disk(sink_position, refined_s , (float(disk_size), "au"), (max(float(0.25*disk_size), 15), "au"))
        s_disk = shape_angular_momentum(trial_disk,sink_position,sink_velocity)/yt.YTQuantity(np.linalg.norm(shape_angular_momentum(trial_disk,sink_position,sink_velocity)),h_unit)
        
        if debug_verbose == True:
            print('Angle refinment:',np.arccos(np.dot(refined_s, s_disk)), 'rad') # the angle between two succesive gass spin refinment levels (how much does it change with each iteration)
            print('S DISK ',n,':',refined_s)
        refined_s = s_disk
    print('refinment done')
    return refined_s
# creating the folder to save outputs for a given sink
sink_folder = save_dir+'/sink_'+str(sink_tag)
if rank == 0:
    if os.path.exists(sink_folder) == False:
            os.makedirs(sink_folder)

# paralelise over files
files = sorted(glob.glob(data_directory+"/*/info*.txt"))

files = files[first_one:last_one:freq]
if debug_verbose == True: 
    if rank == 0:
        print('----------------DISK ANALYSIS----------------')
        print('All output files:')
        print(files)
############ PARALLELISATION ###################
files_perrank=int(len(files)/size)
for count,fil in enumerate(files[rank*files_perrank:(rank+1)*files_perrank]):
    if rank == 0: 
        print('Progress: ', count*size,'/',len(files))

    if debug_verbose == True:
         print('File:',fil,' on rank:',rank)
    
    # Filename manipulation and data loading
    nout = fil[-9:].split('.')[0]
    output_folder = sink_folder+'/output_'+str(nout)
    
    s = rsink(nout,datadir=data_directory)

    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)

    pickle_name = output_folder+'/disk_characteristics.pkl'

    if os.path.exists(pickle_name) == True or len(s['x']) <= sink_tag :
        if os.path.exists(pickle_name) == True:
            print(pickle_name+" exists! just passing through")
        if len(s['x']) <= sink_tag:
            print('Given sink had not yet formed! Passing through...')
            if os.path.exists(output_folder+'/not_formed') == False:
                os.makedirs(output_folder+'/not_formed')
            

    else :
        ds = yt.load(fil , units_override = units)   
        primary_position = np.array([s['x'][sink_tag],s['y'][sink_tag],s['z'][sink_tag]])*units['length_unit'].in_units('au')
        primary_velocity = np.array([s['ux'][sink_tag],s['uy'][sink_tag],s['uz'][sink_tag]])*units['velocity_unit']
        m1 = s['m'][sink_tag]
        m1_u = m1*units["mass_unit"].in_units(m_unit)


        t_after_formation = (s['snapshot_time']*units["time_unit"].in_units('kyr') - s['tcreate'][sink_tag]*units["time_unit"].in_units('kyr'))


        gass_spin_refined = recursive_gass_spin_vector_refinment(primary_position,primary_velocity,disk_size=150,refinment_levels=7)



        boxlen = 4 * 3600 * 180 / np.pi
        dcell=boxlen/(ds.domain_dimensions[0]*2**ds.index.max_level)


        rs = np.logspace(np.log10(2*dcell),np.log10(rmax),nshells+1)
        hs = e*rs

        sphere_100 = ds.sphere(primary_position, (100, "au"))
        mass_in_100 = (sphere_100['Density']*sphere_100['cell_volume']).sum().in_units('g')

        dm1 = s['dm'][sink_tag]*units['mass_unit'].in_units('Msun')
        dt1 = (s['snapshot_time'] - s['tflush'])*units['time_unit'].in_units('yr')
        accretion_rate_primary = dm1/dt1
        g=yt.YTQuantity(6.67430e-8,'dyne*cm**2/g**2')
        R_sun = yt.YTQuantity(1, "Rsun")
        luminosity_primary = (0.5*g*s['m'][sink_tag]*units['mass_unit']*accretion_rate_primary/(4*R_sun)).in_units('Lsun')


        
        # data containers
        v_phi_shellaverage = []
        v_r_shellaverage = []
        v_z_shellaverage = []
        v_phi_delta = []
        v_r_delta = []
        v_z_delta = []
        keplerian_v = []
        h_outer = []
        h_inner = []
        m_outer = []
        m_inner = []
        disk_size_condition=[]
        #shell_sphere_mass=[]
        shells = range(nshells)
        for shell in shells:
            #print('Shell: ',shell,'/',nshells)
            #od_tu = datetime.datetime.now()

            outer_disk = ds.disk(primary_position, gass_spin_refined , (float(rs[shell+1]), "au"), (max(float(hs[shell+1]), 15), "au"))
            outer_p_phi, outer_p_r, outer_p_z, outer_p_phi_2, outer_p_r_2, outer_p_z_2, outer_M = substractable_values(outer_disk,primary_position,primary_velocity,gass_spin_refined)
            outer_l = shape_angular_momentum(outer_disk,primary_position,primary_velocity)
            h_outer.append(outer_l)
            m_outer.append(outer_M)
            #print('OUTER',substractable_values(outer_disk))
            inner_disk = ds.disk(primary_position, gass_spin_refined, (float(rs[shell]), "au"), (max(float(hs[shell+1]), 15), "au"))
            inner_p_phi, inner_p_r, inner_p_z, inner_p_phi_2, inner_p_r_2, inner_p_z_2, inner_M = substractable_values(inner_disk,primary_position,primary_velocity,gass_spin_refined)
            inner_l = shape_angular_momentum(inner_disk,primary_position,primary_velocity)
            h_inner.append(inner_l)
            m_inner.append(inner_M)
            #print('INNER',substractable_values(inner_disk))

            del inner_disk
            del outer_disk
           #print('outer_p_phi',outer_p_phi,'inner_p_phi',inner_p_phi,'outer_M - inner_M',(outer_M - inner_M))
            average_v_phi = ((outer_p_phi - inner_p_phi)/(outer_M - inner_M)).in_units(v_unit)
            average_v_r = ((outer_p_r - inner_p_r)/(outer_M - inner_M)).in_units(v_unit)
            average_v_z = ((outer_p_z - inner_p_z)/(outer_M - inner_M)).in_units(v_unit)

            v_phi_shellaverage.append(average_v_phi)
            v_r_shellaverage.append(average_v_r)
            v_z_shellaverage.append(average_v_z)

            v_phi_dissipation = np.sqrt( (outer_p_phi_2 - inner_p_phi_2)/(outer_M - inner_M) - average_v_phi**2 ).in_units(v_unit) 
            v_r_dissipation = np.sqrt( (outer_p_r_2 - inner_p_r_2)/(outer_M - inner_M) - average_v_r**2 ).in_units(v_unit)
            v_z_dissipation = np.sqrt( (outer_p_z_2 - inner_p_z_2)/(outer_M - inner_M) - average_v_z**2 ).in_units(v_unit)

            v_phi_delta.append(v_phi_dissipation)
            v_r_delta.append(v_r_dissipation)
            v_z_delta.append(v_z_dissipation)

            keplerian_v_shell = np.sqrt(yt.physical_constants.G*(m1_u+inner_M)/yt.YTQuantity(rs[shell],'au')).in_units(v_unit)
            keplerian_v.append(keplerian_v_shell)

            #shell_sphere_mass.append(m_sphere(primary_position,rs[shell]).value)
            # DISK SIZE ESTIMATE:

            if(average_v_phi>0.75*keplerian_v_shell and v_phi_dissipation < 0.3*average_v_phi and rs[shell]>5):
                disk_size_condition.append(shell)

        if(not disk_size_condition):
            disk_size = 0
        else:
            disk_size=rs[disk_size_condition[-1]+1]
        ################# HERE
        v_phi_shellaverage = yt.YTArray(v_phi_shellaverage)
        v_r_shellaverage = yt.YTArray(v_r_shellaverage)
        v_z_shellaverage = yt.YTArray(v_z_shellaverage)
        v_phi_delta = yt.YTArray(v_phi_delta)
        v_r_delta = yt.YTArray(v_r_delta)
        v_z_delta = yt.YTArray(v_z_delta)
        keplerian_v = yt.YTArray(keplerian_v)
        h_outer = yt.YTArray(h_outer)
        h_inner = yt.YTArray(h_inner)
        m_outer = yt.YTArray(m_outer)
        m_inner = yt.YTArray(m_inner)

        save_units = {'distances': 'au','disk_size':'au', 'v_phi_shellaverage': str(v_phi_shellaverage.units) ,
        'v_phi_delta' : str(v_phi_dissipation.units) ,'v_r_shellaverage': str(v_r_shellaverage.units) ,'v_r_delta' : str(v_r_delta.units),
        'v_z_shellaverage': str(v_z_shellaverage.units) ,'v_phi_delta' : str(v_phi_delta.units), 'keplerian_v' : str(keplerian_v.units),
        'mass in 100 au ':str(mass_in_100.units),'accretion_rate':str(accretion_rate_primary.units),'luminosity':str(luminosity_primary.units),
        'h_outer':str(gass_spin_refined.units),'h_inner':str(gass_spin_refined.units),'gass_spin_refined':str(gass_spin_refined.units),
        't_after_formation':str(t_after_formation.units)}

        pickle_file = {'distances':rs,'disk size':disk_size,
            'v_phi': v_phi_shellaverage.value,
            'v_phi error':v_phi_delta.value,
            'v_r':v_r_shellaverage.value,
            'v_r error':v_r_delta.value,
            'v_z':v_z_shellaverage.value,
            'v_z error':v_z_delta.value,
            'v_k':keplerian_v.value,
            'outer disk h':h_outer.value,
            'inner disk h':h_inner.value,
            'gass spin':gass_spin_refined.value,
            'outer disk m':m_outer.value,
            'inner disk m':m_inner.value,
            'mass_in_100':mass_in_100.value,'accretion_rate_primary':accretion_rate_primary.value,'luminosity_primary':luminosity_primary.value,
            't_after_formation':t_after_formation,'output':nout,'primary sink tag':sink_tag,'secondary sink tag':secondary_sink_tag,'units':save_units,'rsink file':s}
                
#idisk
#m_outer[idisk]
#m_outer[:idisk+1].sum() - m_inner[:idisk+1].sum()
                

        #ADD EFFECTIVE ROCHE RADIUS 
        ''' Title: Approximations to the radii of Roche lobes
            Authors: Eggleton, P. P.
            Journal: Astrophysical Journal, Part 1 (ISSN 0004-637X), vol. 268, May 1, 1983, p. 368, 369.
            Bibliographic Code: 1983ApJ...268..368E '''

        if secondary_sink_tag !=None:
            if len(s['x'])> max(sink_tag,secondary_sink_tag):  # if it is a binary system
        
                secondary_position = np.array([s['x'][secondary_sink_tag],s['y'][secondary_sink_tag],s['z'][secondary_sink_tag]])*units['length_unit'].in_units('au')
                secondary_velocity = np.array([s['ux'][secondary_sink_tag],s['uy'][secondary_sink_tag],s['uz'][secondary_sink_tag]])*units['velocity_unit']

                m2_u = s['m'][secondary_sink_tag]*units['mass_unit'].in_units('Msun')
                binary_com = (m1_u*primary_position + m2_u*secondary_position)/(m1_u+m2_u)
                binary_com_v = (m1_u*primary_velocity + m2_u*secondary_velocity)/(m1_u+m2_u)
                separation = yt.YTArray(np.linalg.norm(primary_position - secondary_position),'au')
                
            
                q = m1_u/m2_u
                roche_r =( 0.49*q**(2/3)/(0.6*q**(2/3)+np.log(1+q**(1/3))) )*separation

                dm2 = s['dm'][secondary_sink_tag]*units['mass_unit']  #secondary accretion rate and luminosity is wrong - change after it is done.
                dt2 = (s['snapshot_time'] - s['tflush'])*units['time_unit']
                accretion_rate_secondary = dm2/dt2
                luminosity_secondary = (0.5*g*s['m'][secondary_sink_tag]*units['mass_unit']*accretion_rate_secondary/(4*R_sun)).in_units('Lsun')

                save_units['binary_com']=str(binary_com.units)
                save_units['separation']=str(separation.units)
                save_units['roche_r']=str(roche_r.units)
                save_units['accretion_rate_secondary']=str(accretion_rate_secondary.units)
                save_units['luminosity_secondary']=str(luminosity_secondary.units)
        
                pickle_file['units'] = save_units
                pickle_file['binary_com']=binary_com.value
                pickle_file['separation']=separation.value
                pickle_file['roche_r']=roche_r.value
                pickle_file['accretion_rate_secondary']=accretion_rate_secondary.value
                pickle_file['luminosity_secondary']=luminosity_secondary.value
    
        
        with open(pickle_name, 'wb') as f:
            pickle.dump(pickle_file, f)
        print('Rank ',rank,': ',pickle_name+' saved!')

'''

binary_system_pickle_dict= {'distances':rs,'disk size':disk_size,'v_phi': v_phi_shellaverage,
            'v_phi error':v_phi_delta,
            'v_r':v_r_shellaverage,
            'v_r error':v_r_delta,
            'v_z':v_z_shellaverage,
            'v_z error':v_z_delta,
            'v_k':keplerian_v,
            'sphere mass':shell_sphere_mass,
            'outer disk l':h_outer,
            'inner disk l':h_inner,
            'gass spin':gass_spin_refined.value,
            'outer disk m':m_outer,
            'inner disk m':m_inner,'t_after_formation':t_after_formation,'binary separation':separation,'roche radius':roche_r,'output':nout,'sink tag':sink_tag,'secondary sink tag':secondary_sink_tag,'units':save_units,'rsink file':s}
        
'''