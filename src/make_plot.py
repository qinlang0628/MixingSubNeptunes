import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times'], 'size': '20'})
rc('text', usetex=True)

def plot_profile(planet, profile='Env'):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    plt.subplots_adjust(wspace=0.3)
    
    if profile == 'Env':
        Rs = np.array(planet.Env.r)
        Ps = np.array(planet.Env.P)
        rhos = np.array(planet.Env.rho)
        Ts = np.array(planet.Env.T)
        Ms = np.array(planet.Env.M)
    
    if profile == 'Mantle':
        Rs = np.array(planet.Mantle.r)
        Ps = np.array(planet.Mantle.P)
        rhos = np.array(planet.Mantle.rho)
        Ts = np.array(planet.Mantle.T)
        Ms = np.array(planet.Mantle.M)
        
    if profile == 'Metal':
        Rs = np.array(planet.Metal.r)
        Ps = np.array(planet.Metal.P)
        rhos = np.array(planet.Metal.rho)
        metal_temperature = [planet.Mantle.T[-1]] * len(planet.Metal.r)
        Ts = metal_temperature
        Ms = np.array(planet.Metal.M)
    
    if profile == 'All':
        Rs = np.array(planet.Env.r + planet.Mantle.r + planet.Metal.r)
        Ps = np.array(planet.Env.P + planet.Mantle.P + planet.Metal.P)
        rhos = np.array(planet.Env.rho + planet.Mantle.rho + planet.Metal.rho)
        metal_temperature = [planet.Mantle.T[-1]] * len(planet.Metal.r)
        Ts = np.array(planet.Env.T + planet.Mantle.T + metal_temperature)
        Ms = np.array(planet.Env.M + planet.Mantle.M + planet.Metal.M)

    # R vs P
    Rs /= 6371000
    Ps /= 1e9 # Pa => GPa
    axes[0].plot(Ps, Rs)
    axes[0].set_title("Pressure")
    axes[0].set_xlabel("GPa")
    axes[0].set_ylabel(r'r ($R_\oplus$)')
    # axes[0].set_ylim(9, 10)
    # axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

    # R vs rho
    axes[1].plot(rhos/1e3, Rs,'-*')
    axes[1].set_title("Density")
    axes[1].set_xlabel(r"$g/cm^3$")
    axes[1].set_ylabel(r'r ($R_\oplus$)')
    # axes[1].set_ylim(9, 10)
    # axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

    # R vs T
    axes[2].plot(Ts, Rs)
    axes[2].set_title("Temperature")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel(r'r ($R_\oplus$)')
    # axes[2].set_ylim(.9, 1)
    # axes[2].set_xlim(1, 1000)

#     # R vs M
#     axes[3].plot(Ms, Rs)
#     axes[3].set_title("R vs M")
#     # axes[3].set_xlabel("M")
#     axes[3].set_ylabel(r'r ($R_\oplus$)')
