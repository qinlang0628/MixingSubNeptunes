from eostable import *
from planet_func import integrate_with_atmos, Planet
from utils import guess_history, eos_tables
from read_EOS import read_EOS_Si
import os, time, pickle
import pandas as pd
import logging


if not os.path.exists("log"):
    os.mkdir("log")
logging.basicConfig(filename="log/{}.log".format(time.asctime()), filemode='w', level=logging.DEBUG)


data_folder = "DirEOS2019"
H_filename = "TABLE_H_TP_v1"
He_filename = "TABLE_HE_TP_v1"
HHe_filename = "TABLEEOS_HHE_TP_Y0.275_v1"
Si_filename = 'EOS_Si.pkl'


# read table from folder
EOS = eos_tables()
EOS_H = pd.read_csv("EOS_tables/EOS_H.csv")
EOS_He = pd.read_csv("EOS_tables/EOS_He.csv")
EOS_HHe = pd.read_csv("EOS_tables/EOS_HHe.csv")
NewEOS = read_EOS_Si(Si_filename)

EOS.set_gas_table(EOS_H, EOS_He, EOS_HHe, NewEOS)




##### test case
M_core=6e24 * 7
f=0           # mass fraction of dissolved gas
f_He=0        # mass fraction of Helium for the gas
f_M=0.325     # mass fraction of metal for the core

R_guess = 1* 10 ** 7.375  # initial guess for the radius (unit in m)
dr_coeff = 0.005          # precision of each step during the integration

T_eff=1000              # temperature at the core-envelope boundary (unit in K)
P_eff = 1e7             # pressure at the core-envelope boundary (unit in Pa)

X = 0.68 / 7
X_He = 0.25

new_planet = Planet(M_core, f_M, X, X_He)
new_planet.set_f_fHe(f, f_He)
setattr(new_planet, "P_eff", P_eff)
setattr(new_planet, "T_eff", T_eff)

R_guess_history = guess_history()
R_guess_max = None
R_guess_min = None

R_guess = 24458382
R_guess_max=None
R_guess_min=None


results = integrate_with_atmos(new_planet, R_guess, dr_coeff, EOS, 
                               R_guess_history, R_guess_max, R_guess_min, fugacity=True)