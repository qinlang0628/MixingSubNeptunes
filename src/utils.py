import pandas as pd
from scipy import interpolate
import numpy as np
import pickle
import logging


def S_curve(NewEOS, Smantle=0.0027):
    '''extract the T-rho-P relation with prescribed S_mantle (unit in MJ/K/kg) from the silicate data
    output:
        df (pd.DataFrame): a dataframe with interpolated T-rho-P relation
    '''
    
    # find the curve
    columns = ["T", "rho", "P", "S"]
    
    mantle_pressure = []
    mantle_temperature = []

    # loop across all densities and extract the values for the requested isentrope
    ForsteriteEOS = NewEOS
    for i in range(0,ForsteriteEOS.ND):
        ind = np.where((ForsteriteEOS.S[:,i] > 0))[0]
        interpfunction = interpolate.interp1d(ForsteriteEOS.S[ind,i],ForsteriteEOS.P[ind,i]) # MJ/K/kg, GPa
        mantle_pressure = np.append(mantle_pressure,interpfunction(Smantle)) # GPa
        interpfunction = interpolate.interp1d(ForsteriteEOS.S[ind,i],ForsteriteEOS.T[ind]) # MJ/K/kg, GPa
        mantle_temperature = np.append(mantle_temperature,interpfunction(Smantle)) # GPa

    mantle_density = ForsteriteEOS.rho # g/cm3
    
    df = pd.DataFrame({"T": mantle_temperature, 
                       "rho": mantle_density, 
                       "P": mantle_pressure,
                       "S": np.ones(ForsteriteEOS.ND) * Smantle})
    
    # unit conversion
    df["rho"] = df["rho"] * 1000 #g/cm^3 --> kg/m^3
    df["P"] = df["P"] * 1e9 # (GPa) ---> Pa
    
    return df


def lookup_boundary(num, vals):
    '''given a number and an array, look up the upper and lower closest values from the array
    input:
        num (number): a number
        vals (np.array): an array of shape (n,)
    '''
    sort = vals.argsort()
    idx = np.searchsorted(vals[sort], num, "right")
    lower = vals[sort][idx - 1]
    try:
        upper = vals[sort][idx]
        return lower, upper
    except: # number on the upper boundary
#         logging.debug('Point on the boundary')
        return lower, lower


def interpolate_curve(v1, df, target, col1):
    '''interpolate the value of a target attribute from the value of another attribute and a 1D Silicate EOS table
    input:
        v1(number): the value of "another" attibute
        df(pd.DataFrame): the Silicate EOS table
        target(str): name of the column in the Silicate EOS table for the target attribute 
        col1(str): name of the column for v1 
    output:
        S(number): the value of target attribute
        out_of_table(bool): indicate whether the interpolation is out of boundary
    '''
    # set a flag of out of boundary
    out_of_table = False
    
    # lookup the closest T in the EOS table
    T_vals = df[col1].values
    T_lower, T_upper = lookup_boundary(v1, T_vals)
    
    # lookup values
    S_lower = df[df[col1]==T_lower][target].values[0]
    S_upper = df[df[col1]==T_upper][target].values[0]
    
    # interpolate along col2, then col1
    # if reach boundary, return the boundary value
    if v1 > T_upper:
        logging.debug("warning: {} reaches upper bound: {}".format(col1, v1))
        out_of_table = True
        v1 = T_upper
    elif v1 < T_lower:
        logging.debug("warning: {} reaches lower bound: {}".format(col1, v1))
        out_of_table = True
        v1 = T_lower
    S = interpolate.interp1d([T_lower, T_upper], [S_lower, S_upper])(v1)
    
    # convert the output from 1d array to a scalar
    return np.asscalar(S), out_of_table

def interpolate_df(v1, v2, df, target="S", col1="T", col2="P"):
    '''interpolate the value of target attribute from the values of another two attributes and 2D EOS data
    input:
        v1, v2(number): the values of "another" two attributes
        df(pd.DataFrame): 2D silicate EOS data
        target(str): name of the column in the Silicate EOS data for the target attribute 
        col1, col2 (str): names of of the column for v1 and v2, respectively
    output:
        S(number): the target attribute value
        out_of_table(bool): indicate whether the interpolation is out of boundary
    '''
    # set a flag of out of boundary
    out_of_table = False
    
    # lookup the closest T and P value in the EOS table
    T_vals = df[col1].values
    T_lower, T_upper = lookup_boundary(v1, T_vals)
    P_vals1 = df[df[col1]==T_lower][col2].values
    P_vals2 = df[df[col1]==T_upper][col2].values
    P1, P2 = lookup_boundary(v2, P_vals1)
    P3, P4 = lookup_boundary(v2, P_vals2)
    
    # lookup values
    S1 = df[(df[col1]==T_lower) & (df[col2] == P1)][target].values[0]
    S2 = df[(df[col1]==T_lower) & (df[col2] == P2)][target].values[0]
    S3 = df[(df[col1]==T_upper) & (df[col2] == P3)][target].values[0]
    S4 = df[(df[col1]==T_upper) & (df[col2] == P4)][target].values[0]
    
    # interpolate along col2, then col1
    if v2 > P2: 
        logging.debug("warning: {} reaches upper bound: {}".format(col2, v2))
        out_of_table = True
        v2 = P2
    elif v2 < P1:
        logging.debug("warning: {} reaches lower bound: {}".format(col2, v2))
        out_of_table = True
        v2 = P1
    if v2 > P4: 
        logging.debug("warning: {} reaches upper bound: {}".format(col2, v2))
        out_of_table = True
        v2 = P4
    elif v2 < P3:
        logging.debug("warning: {} reaches lower bound: {}".format(col2, v2))
        out_of_table = True
        v2 = P3
    S_lower = interpolate.interp1d([P1, P2], [S1, S2])(v2)
    S_upper = interpolate.interp1d([P3, P4], [S3, S4])(v2)
    
    if v1 > T_upper: 
        logging.debug("warning: {} reaches upper bound: {}".format(col1, v1))
        out_of_table = True
        v1 = T_upper
    elif v1 < T_lower:
        logging.debug("warning: {} reaches lower bound: {}".format(col1, v1))
        out_of_table = True
        v1 = T_lower
    S = interpolate.interp1d([T_lower, T_upper], [S_lower, S_upper])(v1)
    
    # convert the output from 1d array to a scalar
    return np.asscalar(S), out_of_table


def interpolate_S(T, P, NewEOS):
    '''we want to get S value of a given (T, P) combination
    input:
        T(number): temperature
        P(number): pressure
        NewEOS(extEOStable): EOS of Si is an object with T(n), rho(m), P(n*m) and S(n*m), 
        data from https://github.com/ststewart/aneos-forsterite-2019/blob/master
    '''
    # find the T value that is closest to T_given
    T_lower, T_upper = lookup_boundary(T, NewEOS.T)
    T_lower_idx = np.where(NewEOS.T == T_lower)[0][0]
    T_upper_idx = np.where(NewEOS.T == T_upper)[0][0]
    
    # convert pressure unit from Pa to GPa
    P = P / 1e9
    
    # for T_lower and T_upper, find the points which P is closest to P_given, add them to a list
    points = []
    for i in [T_lower_idx, T_upper_idx]:
        P_vector = NewEOS.P[i,:]
        S_vector = NewEOS.S[i,:]
        sort = P_vector.argsort()
        j = np.searchsorted(P_vector[sort], P, "right") - 1
        if j == NewEOS.ND - 1: 
            raise Exception('Error occurred at initialization: P too large, out of range')
        points.append([NewEOS.T[i], NewEOS.rho[sort][j], P_vector[sort][j], S_vector[sort][j]])
        points.append([NewEOS.T[i], NewEOS.rho[sort][j+1], P_vector[sort][j+1], S_vector[sort][j+1]])
    
    # interpolate S and rho
    S_lower = interpolate.interp1d([points[0][2], points[1][2]], [points[0][3], points[1][3]])(P)
    S_upper = interpolate.interp1d([points[2][2], points[3][2]], [points[2][3], points[3][3]])(P)
    S = interpolate.interp1d([T_lower, T_upper], [S_lower, S_upper])(T)
    
    # S unit: kJ/K/kg
    return np.asscalar(S)


class value_recorder():
    '''value recorder during the integration'''
    def __init__(self):
        self.r = []
        self.M = []
        self.P = []
        self.rho = []
        self.current_T = None
        self.T = []
        self.out_of_boundary = False
        self.out_of_table = False
        
    def set_T(self, T):
        self.current_T = T
        
    def reset_all(self):
        self.r = []
        self.M = []
        self.P = []
        self.rho = []
        self.current_T = None
        self.T = []
        self.out_of_boundary = False
        self.out_of_table = False
    
    def flag_out_of_table(self, out_of_table):
        if out_of_table:
            self.out_of_table = True
    
    def interpolate_df_w_flag(self, v1, v2, df, target, col1="T", col2="P"):
        target_val, out_of_table = interpolate_df(v1, v2, df, target, col1, col2)
        self.flag_out_of_table(out_of_table)
        return target_val
    
    def interpolate_curve_w_flag(self, v1, df, target, col1):
        target_val, out_of_table = interpolate_curve(v1, df, target, col1)
        self.flag_out_of_table(out_of_table)
        return target_val
    
        
    def sort(self):
        # for the debug use, sort the values according to r
        argsort = np.array(self.r).argsort()
        self.M = list(np.array(self.M)[argsort])
        self.P = list(np.array(self.P)[argsort])
        self.rho = list(np.array(self.rho)[argsort])
        self.T = list(np.array(self.T)[argsort])
        self.current_T = self.T[-1]
        
    def cut_r(self, r):
        idx = np.searchsorted(self.r[::-1], r, "left")
        if idx >= 2:
            self.r = self.r[:-idx+1]
            self.M = self.M[:-idx+1]
            self.P = self.P[:-idx+1]
            self.rho = self.rho[:-idx+1]
            self.T = self.T[:-idx+1]
            self.current_T = self.T[-1]
        
class guess_history():
    '''record the history of R_guess'''
    def __init__(self):
        self.r = []
        self.m_error_pair = [] # when R_guess is too small, we will have m error
        self.r_env_boundary_pair = [] #(R_guess, R_env_boundary)
        
        
class eos_tables():
    def __init__(self):
        pass
    
    def set_gas_table(self, EOS_H, EOS_He, EOS_HHe, EOS_Si_obj):
        self.EOS_H = EOS_H
        self.EOS_He = EOS_He
        self.EOS_HHe = EOS_HHe
        self.EOS_Si_obj = EOS_Si_obj
    
    def get_EOS_Si(self, S_Si):
        self.S_Si = S_Si
        self.EOS_Si = S_curve(self.EOS_Si_obj, S_Si) 