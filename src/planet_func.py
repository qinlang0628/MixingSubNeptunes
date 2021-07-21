from scipy.integrate import solve_ivp
from scipy.constants import gravitational_constant, Boltzmann
from utils import *
from formula import *
import logging

class Planet():
    def __init__(self, M_core, f_M, X, X_He):
        self.M_core = M_core
        self.f_M = f_M
        
        # calculate the component mass
        self.M_silicate = (1 - f_M) * M_core
        self.M_metal = f_M * M_core
        
        # total gas mass
        self.X = X
        self.X_He = X_He
        self.M_gas = X * self.M_core
        self.M_gas_He = X_He * self.M_gas
        self.M_gas_H = self.M_gas - self.M_gas_He

        self.flag_out_of_table = False
        self.Env = value_recorder()
        self.Mantle = value_recorder()
        self.Metal = value_recorder()
        
    def reset(self):
        self.Env.reset_all()
        self.Mantle.reset_all()
        self.Metal.reset_all()     
    

    def set_f_fHe(self, f, f_He):
        '''update at the end of each loop'''
        self.f = f
        self.f_He = f_He    # note that f_He is the Helium fraction in the dissolved gas
        
        self.M_mantle = self.M_silicate * 1 / (1 - self.f)
        
        self.M_dissolved = self.M_silicate * self.f / (1 - self.f)
        self.M_dissolved_He = self.M_dissolved * self.f_He
        self.M_dissolved_H = self.M_dissolved * (1 - self.f_He)
        
        self.M_env = self.M_gas - self.M_dissolved
        self.M_env_He = self.M_gas_He - self.M_dissolved_He
        self.M_env_H = self.M_gas_H - self.M_dissolved_H
        
        if self.M_env != 0:
            self.f_env_He = self.M_env_He / self.M_env
        else:
            self.f_env_He = 0
        
        # check if the value is larger than zero
        for mass in [self.M_mantle, self.M_dissolved, self.M_dissolved_H, self.M_dissolved_He, 
                     self.M_env, self.M_env_He, self.M_env_H, self.f, self.f_He]:
            if mass < 0:
                raise ValueError(f"Mass update is invalid for f = {self.f}, f_He = {self.f_He}")
        
        self.calc_total_mass()
    
    def calc_total_mass(self):
        self.M_total = self.M_mantle + self.M_metal + self.M_env
        
    def calc_gas_solubility(self, P_EC, T_EC, EOS):
        if self.X_He == 0: gas = 'H2'
        elif self.X_He == 1: gas = 'He'
        else: gas = 'mix'
        f, f_He = calc_gas_solubility(P_EC, T_EC, EOS.EOS_H, EOS.EOS_He, gas=gas, f_He_env=self.X_He)
        return f, f_He
    
    
    
def func_mantle(r, y, EOS, planet):
    '''
    calculate dMdr and dPdr at one layer in the mantle
    input: 
        r (num): radius
        y ([M, P]): a list which contains mass, pressure
        EOS_Si, EOS_H, EOS_He (pd.DataFrame): the EOS table of materials
        M_boundary (number): the mass boundary to switch from mantle to iron core
        S_Si: the constant entropy of silicate
        f: dissolved gas mass fraction
        f_He: He mass fraction
        ValueRecorder (object): a small object to record values in the integration process
    output:
        (dM_dr, dP_dr): a list which contains the differential values
    '''
    # get P, T, M
    M, P = y[0], y[1]
    f, f_He = planet.f, planet.f_He
    T = planet.Mantle.current_T
    G = gravitational_constant
    
    # record values
    planet.Mantle.r.append(r)
    planet.Mantle.M.append(M)
    planet.Mantle.P.append(P)
    planet.Mantle.T.append(T)
                
    # lookup table to acquire the density of silicate
    rho_Si = planet.Mantle.interpolate_curve_w_flag(P, EOS.EOS_Si, target="rho", col1="P")
        
    # initialize the density of H, He
    rho_H, rho_He = 0, 0
    
    if f_He == 0.275:     # cosmogenic composition, directly apply the density of mixture to rho_H
        rho_H = planet.Mantle.interpolate_df_w_flag(T, P, EOS.EOS_HHe, target="rho")
        rho_He = 0
    else:    
        if f > 0:
            rho_H = planet.Mantle.interpolate_df_w_flag(T, P, EOS.EOS_H, target="rho") 
        if f_He > 0:
            rho_He = planet.Mantle.interpolate_df_w_flag(T, P, EOS.EOS_He, target="rho")
    
    # calculate rho
    rho = calc_rho(rho_Si, rho_H, rho_He, f, f_He)  
    
    dM_dr = 4 * np.pi * r **2 * rho
    dP_dr = - G * M * rho / r ** 2
    
    # record the values
    planet.Mantle.current_T = planet.Mantle.interpolate_curve_w_flag(P, EOS.EOS_Si, target="T", col1="P")
    planet.Mantle.rho.append(rho)
    
    return (dM_dr, dP_dr)


def func_metal(r, y, planet):
    '''
    input: 
        r (num): radius
        y ([M, P]): a list which contains mass, pressure
        EOS_Si, EOS_H, EOS_He (pd.DataFrame): the EOS table of the elements
        M_boundary (number): the mass boundary of silicate and metal
        S_Si: the constant entropy of silicate
        f: dissolved gas mass fraction
        f_He: He mass fraction
        ValueRecorder (object): a small object to record values in the integration process
    output:
        (dM_dr, dP_dr): a list which contains the partial differential values
    '''
    # get P, T, M
    M, P = y[0], y[1]
    G = gravitational_constant
    
    # record values
    planet.Metal.r.append(r)
    planet.Metal.M.append(M)
    planet.Metal.P.append(P)
    
    # lookup table
    rho = calc_rho_Fe(P)  
    
    dM_dr = 4 * np.pi * r **2 * rho
    dP_dr = - G * M * rho / r ** 2
    
    # record the values
    planet.Metal.rho.append(rho)
    
    return (dM_dr, dP_dr)


def func_env(r, y, EOS, planet):
    '''
    calculate dMdr and dPdr at one layer in the mantle
    input: 
        r (num): radius
        y ([M, P]): a list which contains mass, pressure
        EOS_H, EOS_He (pd.DataFrame): the EOS table of materials
        M_boundary (number): the mass boundary to switch from mantle to iron core
        f_He: He mass fraction at each layer
        T_eff: upper boundary condition
        ValueRecorder (object): a small object to record values in the integration process
    output:
        (dM_dr, dP_dr, dT_dr): a list which contains the differential values
    '''
    # get P, T, M
    M, P, T = y[0], y[1], y[2]
    G = gravitational_constant
    k = Boltzmann
    
    # record values
    planet.Env.r.append(r)
    planet.Env.M.append(M)
    planet.Env.P.append(P)
    planet.Env.T.append(T)
        
    # initialize the density of H, He
    rho_H, rho_He = 0, 0
    f, f_He = 1, planet.f_env_He
    
    if f_He == 0.275:     # cosmogenic composition, directly apply the density of mixture to rho_H
        rho_H = planet.Env.interpolate_df_w_flag(T, P, EOS.EOS_HHe, target="rho")
        rho_He = 0
    else:    
        if f > 0:         # pure H
            rho_H = planet.Env.interpolate_df_w_flag(T, P, EOS.EOS_H, target="rho") 
        if f_He > 0:      # pure He
            rho_He = planet.Env.interpolate_df_w_flag(T, P, EOS.EOS_He, target="rho")
       
    # calculate rho
    rho = calc_rho_env(rho_H, rho_He, f_He)  
    
    dM_dr = 4 * np.pi * r **2 * rho
    dP_dr = - G * M * rho / r ** 2
    dT_dr, planet = calc_dT_dr(r, M, T, P,dP_dr, EOS, planet, plan="B")
#     dT_dr, planet = calc_dT_dr(r, M, T, P, EOS, planet)
    
    # record the values
    planet.Env.rho.append(rho)
    
    logging.debug('P, T, rho, dM_dr, dP_dr, dT_dr, rho_H: {}, {}, {}, {}, {}, {}, {}'.format(P, T, rho, dM_dr, dP_dr, dT_dr, rho_H)) # ======debugging purpose
    
    return (dM_dr, dP_dr, dT_dr)


def adjust_boundary(ValueRecorder, M_boundary, EOS_Si):
    # adjust the boundary to the accurate r value (larger than the current r)

    delta_M_boundary = M_boundary - ValueRecorder.M[-1]
    rho_boundary = ValueRecorder.rho[-1]
    delta_r_boundary = calc_delta_r(delta_M_boundary, rho_boundary, ValueRecorder.r[-1])
    r_boundary = ValueRecorder.r[-1] + delta_r_boundary   # delta_r_boundary is a positive number
    dP_dr = - gravitational_constant * M_boundary * rho_boundary / ValueRecorder.r[-1] ** 2
    P_boundary = ValueRecorder.P[-1] + dP_dr * delta_r_boundary   # by definition
    
    T_boundary = ValueRecorder.interpolate_curve_w_flag(P_boundary, EOS_Si, target="T", col1="P")
    rho_boundary = ValueRecorder.interpolate_curve_w_flag(P_boundary, EOS_Si, target="rho", col1="P")

    # save the boundary values
    ValueRecorder.r[-1] = r_boundary
    ValueRecorder.M[-1] = M_boundary
    ValueRecorder.P[-1] = P_boundary
    ValueRecorder.T[-1] = T_boundary
    ValueRecorder.rho[-1] = rho_boundary

    logging.info("boundary (r, M, P, T, rho) after adjustment: {}, {}, {}, {}, {}".format(ValueRecorder.r[-1], \
        ValueRecorder.M[-1], ValueRecorder.P[-1], ValueRecorder.T[-1], ValueRecorder.rho[-1]))
    
    return ValueRecorder


def adjust_boundary_env(planet, M_boundary, EOS):
    # adjust the boundary to the accurate r value (larger than the current r)
    ValueRecorder = planet.Env
    
    delta_M_boundary = M_boundary - ValueRecorder.M[-1]
    rho_boundary = ValueRecorder.rho[-1]
    delta_r_boundary = calc_delta_r(delta_M_boundary, rho_boundary, ValueRecorder.r[-1])
    r_boundary = ValueRecorder.r[-1] + delta_r_boundary   # delta_r_boundary is a positive number
    dP_dr = - gravitational_constant * M_boundary * rho_boundary / ValueRecorder.r[-1] ** 2
    P_boundary = ValueRecorder.P[-1] + dP_dr * delta_r_boundary   # by definition
    
    # calculate T_boundary
    dT_dr, planet = calc_dT_dr(ValueRecorder.r[-1], ValueRecorder.M[-1], ValueRecorder.T[-1], ValueRecorder.P[-1], 
                               dP_dr, EOS, planet, plan="B")
    T_boundary = ValueRecorder.T[-1] + dT_dr * delta_r_boundary
       
    # calculate rho_boundary
    rho_H = planet.Env.interpolate_df_w_flag(T_boundary, P_boundary, EOS.EOS_H, target="rho") 
    rho_He = planet.Env.interpolate_df_w_flag(T_boundary, P_boundary, EOS.EOS_He, target="rho") 
    rho_boundary = calc_rho_env(rho_H, rho_He, planet.f_He) 

    # save the boundary values
    ValueRecorder = planet.Env
    ValueRecorder.r[-1] = r_boundary
    ValueRecorder.M[-1] = M_boundary
    ValueRecorder.P[-1] = P_boundary
    ValueRecorder.T[-1] = T_boundary
    ValueRecorder.rho[-1] = rho_boundary

    logging.info("boundary (r, M, P, T, rho) after adjustment: {}, {}, {}, {}, {}".format(ValueRecorder.r[-1], \
        ValueRecorder.M[-1], ValueRecorder.P[-1], ValueRecorder.T[-1], ValueRecorder.rho[-1]))
    
    return ValueRecorder


def evaluate_boundary(ValueRecorder, M_boundary, R_guess, R_guess_max, R_guess_min):
    break_ = False
    
    # process the result
    logging.info("boundary (r, M, P, T, rho) before adjustment: {}, {}, {}, {}, {}".format(ValueRecorder.r[-1], \
        ValueRecorder.M[-1], ValueRecorder.P[-1], ValueRecorder.T[-1], ValueRecorder.rho[-1]))

    # when R_guess is too large, M could be below zero
    if ValueRecorder.M[-1] < 0:
        logging.error("R_guess is too large. Mass reaches zero at the boundary.")
        R_guess_max = R_guess # decrease R_guess  
        break_ = True

    # ??? when R_guess is so small that the integration ends when r reaches 0
    if ValueRecorder.M[-1] > M_boundary:
        logging.error("R_guess is too small. Integration ends before the boundary.")
        R_guess_min = R_guess # increase R_guess
        break_ = True

    if ValueRecorder.M[-1] < M_boundary:
        pass
    
    return break_, R_guess_max, R_guess_min


def calc_R(planet, R_guess, dr_coeff, EOS, R_guess_history,
      R_guess_max = None, R_guess_min = None, 
      uniform=True, T_jump=False):
    '''
    input:
        M_core: mass of core 
        X: envelope mass / core mass # new
        X_He: Helium mass / envelope mass # new
        f_M: metal mass / core mass
        R_guess: initial guess of radius
        T_eff, P_eff: upper boundary condition # new
        dr_coeff: a coeffcient to set the maxium possible dr for the RK4 integrator
        uniform: uniform dissolved gas assumption
        T_jump: whether there is a temperature jump between the magma ocean and iron core
        EOS_H, EOS_He: input tables of H and He EOS
        EnvValueRecorder, ValueRecorder, MetalValueRecorder: value recorder of the integration values (for debugging purpose)
        EOS_Si_obj: input object of silicate EOS
    '''
        # append to guess history
    logging.debug('Start calculating R')
    R_guess_history.r.append(R_guess)
    
    R_earth = 6371000
    # define the range of R_guess to from 0.5*R_earth to 10*R_earth
    if not R_guess_max:
        R_guess_max = R_guess * 1e1
    if not R_guess_min:
        R_guess_min = R_guess * 0.5
        
    planet.reset()
    
    # integration details
    dr_max = dr_coeff * R_guess # set the max possible dr for the RK4 integrator
    r_lim = 5 # stop integration when r is smaller than 5 * dr
    
    # get constants
    G = gravitational_constant
    
    while True:
        # atmostpheric integration
        try: 
            def reach_boundary(r, y, *args):
                return y[0] - (planet.M_total - planet.M_env) # when mass reaches the iron core boundary, stop integration
            reach_boundary.terminal = True

            sol = solve_ivp(func_env, 
                    t_span=(R_guess, r_lim * abs(dr_max)), # integrate from R_guess to 0
                    y0=(planet.M_total, planet.P_eff, planet.T_eff), # initial y
                    method='RK45', # Explicit Runge-Kutta method of order 5(4)
                    max_step=abs(dr_max), # maximum stepsize is dr
                    events = [reach_boundary],
                    args=[EOS, planet] # pass in additional args
                    )
        except:
            raise Exception("Integration fails for envelope.")

        break_, R_guess_max, R_guess_min = evaluate_boundary(planet.Env, planet.M_total - planet.M_env, R_guess, R_guess_max, R_guess_min)
        
        if break_:            
            print("Break @ Env")
            R_guess_history.r_env_boundary_pair.append((R_guess, None)) # ===== debugging purpose
            break
        
        planet.Env.cut_r(sol.t_events[0][0])
        planet.Env = adjust_boundary_env(planet, planet.M_total - planet.M_env, EOS)

        # get T_EC, P_EC for mantle
        T_EC = planet.Env.T[-1]
        P_EC = planet.Env.P[-1]
        print("T_EC = {:.2f} K, P_EC = {:.2f} GPa".format(T_EC, P_EC/1e9))

        # for silicate, get S from T and P at the envelope-mantle boundary
        # then interpolate the EOS_Si with a fixed entropy S_Si
        S_Si = interpolate_S(T_EC, P_EC, EOS.EOS_Si_obj)
        EOS.get_EOS_Si(S_Si)

        # interpolate other initial conditions
        rho_Si = planet.Mantle.interpolate_curve_w_flag(P_EC, EOS.EOS_Si, target="rho", col1="P")
        rho_H = planet.Mantle.interpolate_df_w_flag(T_EC, P_EC, EOS.EOS_H, target="rho")    
        rho_He = planet.Mantle.interpolate_df_w_flag(T_EC, P_EC, EOS.EOS_He, target="rho")

        # calculate the density of mixture at the envelope-mantle boundary
        rho = calc_rho(rho_Si, rho_H, rho_He, planet.f, planet.f_He)

        try: 
            # integration over the mantle
            planet.Mantle.set_T(T_EC)
            
            def reach_boundary(r, y, *args):
                return y[0] - planet.M_metal # when mass reaches the iron core boundary, stop integration
            reach_boundary.terminal = True

            sol = solve_ivp(func_mantle, 
                    t_span=(planet.Env.r[-1], r_lim * abs(dr_max)), # integrate from R_guess to 0
                    y0=(planet.Env.M[-1], P_EC), # initial y
                    method='RK45', # Explicit Runge-Kutta method of order 5(4)
                    max_step=abs(dr_max), # maximum stepsize is dr
                    events = [reach_boundary],
                    args=[EOS, planet] # pass in additional args
                    )
            
        except:
            raise Exception("Integration fails for mantle.")


        # process_integration result
        break_, R_guess_max, R_guess_min = evaluate_boundary(planet.Mantle, planet.M_metal, R_guess, R_guess_max, R_guess_min)
        
        R_guess_history.r_env_boundary_pair.append((R_guess, planet.Env.r[-1])) # ===== debugging purpose
        if break_:
            print("Break @ Mantle")
            break
        
        try:
            planet.Mantle.cut_r(sol.t_events[0][0])
        except:
            print(sol.t_events)
            return R_guess, planet
        planet.Mantle = adjust_boundary(planet.Mantle, planet.M_metal, EOS.EOS_Si)

        # integration for metal core
        try:
            def reach_boundary(r, y, *args):
                return y[0] # when mass = 0
            reach_boundary.terminal = True

            solve_ivp(func_metal, 
                    t_span=(planet.Mantle.r[-1], r_lim * abs(dr_max)), # integrate from R_guess to 0
                    y0=(planet.M_metal, planet.Mantle.P[-1]), # initial y
                    method='RK45', # Explicit Runge-Kutta method of order 5(4)
                    max_step=abs(dr_max), # maximum stepsize is dr
                    events = [reach_boundary],
                    args=[planet] # pass in additional args
                    )
        except:
            logging.error("Integration for metal ends at r = {}".format(planet.Metal.r[-1]))
            raise Exception("Integration fails for metal.")

        # post integration
        last_r = planet.Metal.r[-1]
        if last_r > 0:
            # if the last radius is small enough
            # calculate the mass for the last bit of metal component
            planet.Metal.M.append(planet.Metal.M[-1] - \
                                        calc_M_Fe(planet.Metal.rho[-1], planet.Metal.r[-1]))
            planet.Metal.r.append(0)
            planet.Metal.P.append(planet.Metal.P[-1])
            planet.Metal.rho.append(planet.Metal.rho[-1])

        # evaluate the error of integration result
        last_layer_M = planet.Metal.M[-1]
        error_ratio = last_layer_M / planet.M_core

        print("M_error / M_core: {:.2f}%".format(error_ratio * 100))
        logging.info("M_error / M_core: {:.2f}%".format(error_ratio * 100))
                
        if abs(error_ratio) <= 0.01:
            if planet.Mantle.out_of_table or planet.Metal.out_of_table or planet.Env.out_of_table:
                logging.error("Integration finishes out of table, at R = {}".format(R_guess))
                raise Exception("Integration finishes out of table, at R = {}".format(R_guess))
            else:
                logging.info("Integration finishes at {}".format(R_guess))
                return R_guess, planet #====== debugging purpose

        elif error_ratio > 0.001:
            R_guess_min = R_guess # increase R_guess
            R_guess_history.m_error_pair.append((R_guess, error_ratio))
            break
        else:
            R_guess_max = R_guess # decrease R_guess  
            R_guess_history.m_error_pair.append((R_guess, error_ratio))
            break

    # if R_guess needs to be adjusted and re-integrated
    R_guess_next = np.sqrt(R_guess_max * R_guess_min)        
    logging.info("\nnew_R_guess: {:.3f} R_E".format(R_guess_next / 6371000))
    print("\nnew_R_guess: {:.3f} R_E".format(R_guess_next / 6371000))

    return calc_R(planet, R_guess_next, 
                  dr_coeff, EOS,
                  R_guess_history,
                  R_guess_max, R_guess_min, 
                  uniform=True, T_jump=False)



def integrate_with_atmos(planet, R_guess, dr_coeff, EOS, R_guess_history,
      R_guess_max = None, R_guess_min = None, 
      uniform=True, T_jump=False, fugacity=False):

    R_guess, planet = calc_R(planet, R_guess, dr_coeff, EOS, R_guess_history,
                             R_guess_max = R_guess_max, R_guess_min = R_guess_min, uniform=True, T_jump=False)
    
    if not fugacity:
        return R_guess, planet
    else:
        f, f_He = calc_F(planet, EOS)
        f, f_He = adjust_F(planet, f, f_He)
        
        # added by Bowen (07072021)
        x_H = f * (1 -  f_He)
        x_He = f * f_He
        R_last = R_guess
        
        planet.set_f_fHe(f, f_He)
        
        while True:
            # re-calculate R, f
            R_guess, planet = calc_R(planet, R_guess, dr_coeff, EOS, R_guess_history,
                                     R_guess_max = None, R_guess_min = None, uniform=True, T_jump=False)
            f, f_He = calc_F(planet, EOS)
            
            x_H = f * (1 -  f_He)
            x_He = f * f_He
            
            R_ratio = (R_last - R_guess) / R_last
            
            R_E = 6371000
            # compare new and previous f, f_He
            # logging.info("x_H_ratio = {}, x_H_last = {}, x_H = {}, x_He_ratio = {}, x_He_last = {}, x_He = {} "\
            #            .format(x_H_ratio, x_H_last, x_H, x_He_ratio, x_He_last, x_He))
            logging.info("r_ratio = {} R_E, R_last = {} R_E, R_guess = {} R_E,"\
                         .format(R_ratio / R_E, R_last / R_E, R_guess / R_E))
            
            if abs(R_ratio * 100) > 0.1:
                # obtain the solubility for the next round
                f, f_He = adjust_F(planet, f, f_He)
                x_H = f * (1 -  f_He)
                x_He = f * f_He
        
                # re-integrate
                R_last = R_guess
                
                planet.set_f_fHe(f, f_He)
            else:
                print("Congratulations! The itration is successful! :)")
                break
                
    return R_guess, planet