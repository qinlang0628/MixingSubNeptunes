import numpy as np
from scipy.constants import gravitational_constant, Boltzmann
from utils import *
import logging

def log10_linspace(start, stop, num):
    '''create a P array of shape (num,), in the interval [start, stop], with points distributed evenly on log10 scale'''
    log_start = np.log10(start)
    log_stop = np.log10(stop)
    log_array = np.linspace(log_start, log_stop, num)
    array = 10 ** log_array
    return array

def calc_fugacity_coeff(T_b, P_b, EOS_table, gas="H2"):
    ideal_gas_constant = 8.314 # unit in J/mol/K
    num_points = 5000 # number of points for integration
    lower_pressure = 1e6   # lower bound for integration, unit in Pa (1e5 is also fine, but smaller than that lead to large error)
    
    if gas == "H2":
        mol_mass = 0.002 # unit in kg/mol
    elif gas == "He":
        mol_mass = 0.004 # unit in kg/mol
    
    P_array = log10_linspace(lower_pressure, P_b, num_points)
    rho_array = np.zeros_like(P_array)
    
    for (i, P) in enumerate(P_array):
        rho_array[i], out_of_table = interpolate_df(T_b, P, EOS_table, target="rho", col1="T", col2="P")
        if out_of_table:
            raise ValueError("(P, T) is out of table")
    
    # density if it's ideal gas        
    rho_ideal_array = P_array * mol_mass  / (ideal_gas_constant * T_b)
    # degree of compression
    z_array = rho_ideal_array / rho_array
    # (z-1) / p
    partition_function = (z_array - 1) / P_array
    # phi: fugacity coefficient
    ln_phi = np.sum(partition_function[:-1] * np.diff(P_array))
    phi = np.exp(ln_phi)
    
    return phi

def calc_f(P_b, T_b, phi):
    '''calculate solubility (mass fraction), giving the partial pressure, temperature and fugacity coefficient'''
    T0 = 4000     # unit in K
    return 1e-11 * P_b * phi * np.exp(- T0 / T_b)
    
def calc_gas_solubility(P_b, T_b, EOS_H, EOS_He, gas, f_He_env=None):
    '''
    f: mass fraction of dissolved gas in the mantle
    f_He: mass fraction of helium for the dissolved gas
    f_He_env: mass fraction of helium in the envelope
    '''
    if gas == "H2":
        phi = calc_fugacity_coeff(T_b, P_b, EOS_H, gas)
        f = calc_f(P_b, T_b, phi)
        f_He = 0
        
    elif gas == "He":
        phi = calc_fugacity_coeff(T_b, P_b, EOS_He, gas)
        f = calc_f(P_b, T_b, phi)
        f_He = 1
        
    elif gas == "mix":
        if f_He_env is not None:
            P_He = P_b * f_He_env # partial pressure of He
            P_H2 = P_b * (1 - f_He_env) # partial pressure of H2
            phi_H2 = calc_fugacity_coeff(T_b, P_b, EOS_H, "H2")
            phi_He = calc_fugacity_coeff(T_b, P_b, EOS_He, "He")
            x_H2 = calc_f(P_H2, T_b, phi_H2)    # dissolved mass fraction of H2
            x_He = calc_f(P_He, T_b, phi_He)    # dissolved mass fraction of H2
            f = x_H2 + x_He                     # total dissolved mass fraction
            f_He = x_He / f                     # Helium fraction in the dissolved gas
        else:
            raise Exception("Please input mass fraction of helium in the envelope (0<f<1)")
        
    else:
        raise Exception("Please input a correct gas name (H2, He or mix)")
    
    return f, f_He


# integration
def calc_rho(rho_Si, rho_H, rho_He, f, f_He):
    '''calculate the density of silicate, hydrogen and helium mixture, using the linear mixing approximation'''
    if f > 0 and f_He > 0:
        if f_He == 0.275:    # cosmogonic He mass fraction
            return 1.0 / ( (1-f)/rho_Si + f/rho_H )
        else:                # Pure He
            return 1.0 / ( (1-f)/rho_Si + f*(1-f_He)/rho_H + f*f_He/rho_He )
    elif f > 0:              # Pure H
        return 1.0 / ( (1-f)/rho_Si + f/rho_H )
    else:                    # No dissolved gas
        return rho_Si
    
def calc_rho_env(rho_H, rho_He, f_He):
    '''calculate the density of hydrogen and helium mixture, using the linear mixing approximation'''
    if f_He > 0:
        return 1.0 / ( (1-f_He)/rho_H + f_He/rho_He )
    else:
        return rho_H

def calc_c_s(k, T, mu):
    '''calculate sound of speed, input is Boltzmann constant, temperature, mu'''
    m_H = 1.00784 * 1.66 * 1e-27
    return (k * T / mu / m_H) ** (1/2)


def calc_dT_dr(r, M, T, P, dP_dr, EOS, planet, plan="B"):
    ''''''
    if plan == "A":
        G = gravitational_constant
        k = Boltzmann
        f_He = planet.f_env_He
        mu = 1 / ((1 - f_He) / 2 + f_He / 4)
        c_s = calc_c_s(k, planet.T_eff, mu)
        lambda_ad_H = planet.Env.interpolate_df_w_flag(T, P, EOS.EOS_H, target="grad_ad")
        lambda_ad_He = planet.Env.interpolate_df_w_flag(T, P, EOS.EOS_He, target="grad_ad")

        gamma_H = 1 / (1 - lambda_ad_H)
        gamma_He = 1 / (1 - lambda_ad_He)
        gamma_mix = ((1 - f_He) * gamma_H / (gamma_H - 1) + f_He * gamma_He / (gamma_He - 1)) \
                / ((1 - f_He) / (gamma_H - 1) + f_He / (gamma_He - 1))
        lambda_ad_mix = (gamma_mix - 1 ) / gamma_mix

        dT_dr = - G * M * T * lambda_ad_mix / r ** 2 / c_s ** 2
        return dT_dr, planet

    else:
        G = gravitational_constant
        k = Boltzmann
        f_He = planet.f_env_He

        lambda_ad_H = planet.Env.interpolate_df_w_flag(T, P, EOS.EOS_H, target="grad_ad")
        lambda_ad_He = planet.Env.interpolate_df_w_flag(T, P, EOS.EOS_He, target="grad_ad")

        gamma_H = 1 / (1 - lambda_ad_H)
        gamma_He = 1 / (1 - lambda_ad_He)
        gamma_mix = ((1 - f_He) * gamma_H / (gamma_H - 1) + f_He * gamma_He / (gamma_He - 1)) \
                / ((1 - f_He) / (gamma_H - 1) + f_He / (gamma_He - 1))
        lambda_ad_mix = (gamma_mix - 1 ) / gamma_mix

        dT_dr = T * lambda_ad_mix / P * dP_dr
        return dT_dr, planet

def calc_delta_r(delta_M, rho, R):
    '''calculate the thickness of the last layer in the mantle'''
    return delta_M / (4 * np.pi * rho * R ** 2)
        
def calc_rho_Fe(P, rho_Fe_0=8300.00, c=0.00349, n=0.528):
    '''calculate the density of iron core
    formula from Table 3: https://arxiv.org/pdf/0707.2895.pdf'''
    return rho_Fe_0 + c * P ** n

def calc_M_Fe(rho_Fe, R):
    '''calculate the mass of the last layer (actually a sphere) of iron core'''
    return (4/3) * np.pi * R ** 3 * rho_Fe


def calc_F(planet, EOS):
    # calculate gas solubility
    T_EC = planet.Env.T[-1]
    P_EC = planet.Env.P[-1]
    f, f_He = planet.calc_gas_solubility(P_EC, T_EC, EOS)
    print("P_EC, T_EC, f, f_He, ", P_EC, T_EC, f, f_He)
    return f, f_He
    
    
def adjust_F(planet, f, f_He, damping_factor=0.95):
    '''adjust the Hydrogen and Heluim solubility based on availability and total dissolved mass fraction
    input:
        damping (float): apply numeric damping. If damping_factor=1, do not apply damping
    '''
       
    if f > 0.5: 
        logging.debug('warning: estimated dissolved mass fraction {} > 50%, \
        adjust it toward to 50% instead'.format(f))
        f = 0.5
        
    # one possible limit: all gas dissolved into the magma layer
    x_H = f * (1 -  f_He)
    x_He = f * f_He
    x_H_limit = planet.M_gas_H / (planet.M_gas_H + planet.M_gas_He + planet.M_silicate) 
    x_He_limit = planet.M_gas_He / (planet.M_gas_He + planet.M_gas_He + planet.M_silicate)
    if x_H > x_H_limit or x_He > x_He_limit:
        if x_H > x_H_limit and x_He > x_He_limit:
            logging.debug('warning: estimated dissolved H2 and He fraction too high, \
            adjust it toward to all gas dissolved instead.')
            x_H = x_H_limit
            x_He = x_He_limit
        elif x_H > x_H_limit:
            logging.debug('warning: estimated dissolved H2 fraction {} too high, \
            adjust it toward to all H2 dissolved instead.'.format(x_H))
            x_H = planet.M_gas_H * (1 - x_He) / (planet.M_gas_H + planet.M_silicate)
        if x_He > x_He_limit:
            logging.debug('warning: estimated dissolved He fraction {} too high, \
            adjust it toward to all He dissolved instead.'.format(x_He))
            x_He = planet.M_gas_He * (1 - x_H) / (planet.M_gas_He + planet.M_silicate)
        f = x_H + x_He
        f_He = x_He / f
    
    # apply damping
    last_f = planet.f
    last_f_He = planet.f_He
    f = last_f + (f - last_f) * damping_factor
    if x_H != 0 and x_He != 0:
        f_He = last_f_He + (f_He - last_f_He) * damping_factor
        
    return f, f_He