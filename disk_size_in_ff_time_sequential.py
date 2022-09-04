
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyramses import rsink
import glob
import yt 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sink", help="Sink number for which you want the analysis to be done for. First created sink is labeld with 0")
parser.add_argument("-d", "--data", help="Sink data directory")
parser.add_argument("-save", "--save", help="Where to save the output")
parser.add_argument("-primary", "--primary", help="If the sink is a part of a binary, the sink id of the primary ",default=False)
#parser.add_argument("-verbose", "--verbose", help="PRINT MORE OUTPUT",default='False')
#parser.add_argument("-c", "--callculation", help="For yt implementation write yt, for Vitos implementation write V",default='V')
args = parser.parse_args()

start = time.time()

sink_tag = int(args.sink)
data_directory = args.data
save_directory = args.save
primary_sink_tag = int(args.primary)
debug_verbose = True
l_callculation_type = 'yt'
save_plot = 'True'

# # # HARDCODING PARAMETERS # # # 
nshells = 50
rmax = 15000 # in au

h_unit = 'au**2/yr'
m_unit ='Msun'
r_unit ='au'
t_unit = 'yr'


def _Density(field,data):
  #Overwrites density field 
  density_unit = (data.ds.mass_unit/data.ds.length_unit**3).in_cgs()
  density = data[('ramses', 'Density')].value*density_unit
  density_arr = yt.YTArray(density, 'g/cm**3')
  del density
  del density_unit
  return density_arr
yt.add_field('Density', function=_Density, units=r'g/cm**3')



def all_outputs(dat,sorted_filenames):  
    out= []
    nout = [fil[-5:] for fil in sorted_filenames]
    for n in range(len(nout)):
        out.append( rsink(nout[n],datadir=dat) )
    return out

def birth_index(all_rsink,sink):
    # from all the outputs in data folder find the index so that given sink is formed:
    ii = 0
    data_index_where_sink_formed = None
    for ii in range(len(all_rsink)): 
        if len(all_rsink[ii]['m'])  == sink+1:
            data_index_where_sink_formed = ii 
            return data_index_where_sink_formed
    return data_index_where_sink_formed


def shape_dr_dv(yt_shape, com, v_com=yt.YTArray([0,0,0],'km/s')): 
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

def shape_angular_momentum(yt_shape,sink_position,sink_velocity=yt.YTArray([0,0,0],'km/s')): # vito implementation.
    '''Calculate the angular momentum of the gass inside a given shape'''
    dens = yt_shape['Density'].value*units['density_unit']
    dr , dv = shape_dr_dv(yt_shape,sink_position,sink_velocity ) 
    mass = (dens*yt_shape['cell_volume']).sum()
    r_x_v = yt.ucross(dr,dv,axis=0 )
    cell_L = dens*yt_shape['cell_volume']*r_x_v/mass
 
    L_gass = [cell_L[0].sum(),cell_L[1].sum(),cell_L[2].sum()]  #IF THIS DOESNT WORK I AM THE POPE
    #print('Lgass:',L_gass)
    return yt.YTArray(L_gass)


def recursive_gass_sphere_com_refinment_yt(ds,start_position,starting_sphere_size=15000,refinment_levels=20):
    ''' Finds a center of mass of a a gas cloud based on iterativley callculating centers of mass starting with an initial value. First a we callculate the total angular momentum in a sphere centered at start_position and the radius of sphere_size in AU. Then we recompute the center of mass in a sphere centered on the previous callculation of the center of mass. This is done recursevlly for refinment_level times.  '''
   #YT IMPLEMENTATION - SLOW   TO DO: YOUR IMPLEMENTATION
    sphere_center = start_position
    for n in range(refinment_levels):
        sphere = ds.sphere(sphere_center, (starting_sphere_size, "au")) 
        com = sphere.quantities.center_of_mass(use_gas=True, use_particles=False).in_units(r_unit)
        refinment = np.linalg.norm(sphere_center - com)
        if debug_verbose == True:
            print('COM sphere ',n,':',com)
            print('COM refinment:',refinment, r_unit) # the angle between two succesive gass spin refinment levels (how much does it change with each iteration)
        if refinment == 0:
            #starting_sphere_size = starting_sphere_size*9/10
            break
        sphere_center = com    
    return com

def sphere_l_m(data_set,center,radius):
    sphere = data_set.sphere(center, (radius, "au"))
    dens = sphere['Density']
    mass = (dens*sphere['cell_volume']).sum()
    #mass=sphere.quantities.total_mass()
    if l_callculation_type == 'yt':
        h = sphere.quantities.angular_momentum_vector(use_gas=True, use_particles=False)

    elif l_callculation_type == 'vito':
        h = shape_angular_momentum(sphere,center)

    else:
        print('Wrong callculation type spesification (V or yt)')
        return None
    
    return h.in_units(h_unit) , mass.in_units(m_unit)          # yt implementation
                # h not l

def load_pickle(file):
    '''Loads a pickle file simply'''
    information=open(file,"rb")
    result=pickle.load(information,encoding='bytes')
    information.close()
    return result
units = {'length_unit': yt.YTQuantity(4.0, 'pc'),
         'velocity_unit': yt.YTQuantity(0.18, 'km/s'),
         'time_unit': yt.YTQuantity(685706129102738.9, 's'),
         'mass_unit': yt.YTQuantity(3000.0, 'Msun'),
         'density_unit': yt.YTQuantity(46.875, 'Msun/pc**3')}


sorted_files = sorted(glob.glob(data_directory+"/output*"))[5:]
all_rsink_files = all_outputs(data_directory,sorted_files) #s = rsink(all=True,monotonic=True,datadir=data_directory)

date_of_birth = birth_index(all_rsink_files,sink_tag)
birth_timestep = all_rsink_files[date_of_birth]

birth_position = np.array([birth_timestep['x'][sink_tag],birth_timestep['y'][sink_tag],birth_timestep['z'][sink_tag]])*units['length_unit'].in_units('au')

pre_collapse_timestep = all_rsink_files[date_of_birth-1]
pre_collapse_file = glob.glob(sorted_files[date_of_birth-1]+'/info_*')[0]
print('pre collapse file: ',pre_collapse_file)
out_nr = pre_collapse_file[-9:-4]
print(out_nr)
ds = yt.load(pre_collapse_file , units_override = units)
sphere_com = recursive_gass_sphere_com_refinment_yt(ds,birth_position,starting_sphere_size=100,refinment_levels=15)
#sphere_com = yt.YTArray([576628.6414273,58341.24273965,658142.34168544], 'au')
#       birth_position: find sink position on the output where it got created, - FIXED: center on refined sphere_com
#       pre_collapse_timestep: take the second file in the folder - first is always weird! 
#       for spheres with r < 10 000 au :
#           take the pre collapse timestep and center a sphere on sphere_com
#           callculate: l,mass of the onion shells between neighbouring spheres



boxlen = 4 * 3600 * 180 / np.pi #FIND THIS?
dcell=boxlen/(ds.domain_dimensions[0]*2**ds.index.max_level) 

rs = np.logspace(np.log10(16*dcell),np.log10(rmax),nshells+1)
#rs=np.linspace(2*dcell,rmax,nshells+1)
print(' Loading data time: ',time.time() - start,' s')
ls_sphere=yt.YTArray(np.zeros((nshells,3)),h_unit)
ls_mag=yt.YTArray(np.zeros(nshells),h_unit)
m_sphere=yt.YTArray(np.zeros(nshells),m_unit)
l_shell=yt.YTArray(np.zeros((nshells,3)),h_unit)
m_shell=yt.YTArray(np.zeros(nshells),m_unit)
shells = range(nshells)

for shell in shells:
    shell_time = time.time() 
    l,m = sphere_l_m(ds,sphere_com,rs[shell])  
    print(m)
    print(shell)
    ls_sphere[shell]=l.in_units(h_unit)
    m_sphere[shell] = m.in_units(m_unit)
    
    if (shell > 0):
        print(rs[shell])
        print(rs[shell-1])
        print(m_sphere[shell])
        print(m_sphere[shell-1])
        m_shell[shell] = (m_sphere[shell] - m_sphere[shell-1])  ### ALL ZEROSwtf
        if(m_shell[shell]==yt.YTQuantity(0.,'g')):
            print('Shell: ',shell)
            print('Mass of neighbouring spheres is the same. ')
            #time.sleep(100)
            #ls_sphere=np.delete(ls_sphere,shell,0)
            #m_sphere=np.delete(m_sphere,shell,0)  
            #rs=np.delete(rs,shell,0)
            #shells.pop(shell)
        else:
            l_shell[shell] = ((ls_sphere[shell] * m_sphere[shell] - ls_sphere[shell-1] * m_sphere[shell-1]) / m_shell[shell]).in_units(h_unit)
        
    else :
        l_shell[shell] = l.in_units(h_unit)
        m_shell[shell] = m.in_units(m_unit)
    
    #ls_mag[shell]=yt.YTQuantity(np.linalg.norm(l_shell[shell]),h_unit) #PYTHAGOURS?
    ls_mag[shell]= ((l_shell**2).sum())**0.5

rs_yt = yt.YTArray(rs,r_unit)
g_cgs_yt=yt.YTQuantity(6.67430e-8,'dyne*cm**2/g**2')
t_ff_yt = ((9*rs_yt[:-1]**3/(128*g_cgs_yt*m_sphere))**0.5).in_units(t_unit)
r_cf_yt = (ls_mag**2/(g_cgs_yt*m_sphere)).in_units(r_unit)

pickle_file = {'rs':rs_yt,'t_ff':t_ff_yt, 'l_shell_magnitude' : ls_mag ,'l_shells':l_shell ,'r_cf':r_cf_yt,'m_spheres':m_sphere,'m_shells': m_shell, 'l_spheres':ls_sphere,'com_sphere':sphere_com,'nout':out_nr,'sink':sink_tag}

pickle_name = save_directory+'/sink_'+str(sink_tag)+'_out_'+str(out_nr)+'_ff_timeYTMASS.pkl'
#with open(pickle_name, 'wb') as f:
    #pickle.dump(pickle_file, f)
#print(pickle_name+' saved!')

if(save_plot == 'True'):
    fig, ax = plt.subplots(1,1,constrained_layout=True)
    fig.suptitle('Sink '+ str(sink_tag) + ' Output: '+str(out_nr), fontsize=9)

    im = ax.scatter(rs_yt[:-1],r_cf_yt.in_units('au'),c=t_ff_yt.in_units('kyr'),cmap='plasma',s=7,vmin=0, vmax=40)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Shell free-fall time [kyr]', rotation=270,labelpad=11)

    ax.plot(rs_yt[:-1],rs_yt[:-1],'--',label = r'$r_{\mathrm{cf}} = r_{\mathrm{shell}}$',c='orange')
    ax.set(#xlim=(rs[0], rs[-1]),
            ylim=(0, np.max(r_cf_yt)+yt.YTQuantity(5, 'au')),
            #title=f'Rotational velocity per shell',
            xlabel=r'$r_{\mathrm{shell}}$  [AU]' ,
            ylabel=r'$r_{\mathrm{cf}} $  [AU]' 
            )
    ax.set_yscale('linear')

    
    if primary_sink_tag != False:
        primary_position = [pre_collapse_timestep['x'][primary_sink_tag],pre_collapse_timestep['y'][primary_sink_tag],pre_collapse_timestep['z'][primary_sink_tag]]*units['length_unit'].in_units('au')
        distance_to_primary = np.linalg.norm(sphere_com.in_units('au') -primary_position)
        ax.axvline(distance_to_primary,color='grey',label='Distance to primary')
    ax.legend(loc='lower right')        
    fig_save_name = save_directory+'/sink_'+str(sink_tag)+'_out_'+str(out_nr)+'_ff_time_plot100.pdf'
    plt.savefig(fig_save_name,bbox_inches='tight', dpi=350)
    print(str(fig_save_name)+' saved!')

print('Time to run the script: ', time.time() - start)
