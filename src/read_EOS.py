# Data Source:
# "A new equation of state for dense hydrogen-helium mixtures"
# Chabrier, Mazevet & Soubiran, ApJ 2019

import pickle
import pandas as pd
import logging

def read_table(filename):
    '''
    parse the data file for H/He
    input a filename, return a panda dataframe
    '''
    logging.info("Read Table: {}".format(filename))
    
    # read file into pandas dataframe
    with open(filename, 'r') as file:
        file.readline()
        lines = file.readlines()  
        # log T [K]      log P [GPa]   log rho [g/cc] log U [MJ/kg]  log S [MJ/kg/K] 
        # dlrho/dlT_P    dlrho/dlP_T     dlS/dlT_P      dlS/dlP_T       grad_ad
        columns = ["logT", "logP", "logRho", "logU", "logS", 
                   "dlogRho/dlogT", "dlogRho/dlogP", "dlogS/dlogT", "dlogS/dlogP", "grad_ad"]
        df = pd.DataFrame(columns=columns)

        for line in lines:
            line = line.strip().split(" ")
            line = [x for x in line if x != ""]

            # ignore logT
            if len(line) == 2: 
                continue

            # read other attribute
            elif len(line) == 10: 
                line = [float(x) for x in line]
                df = df.append(dict(zip(columns, line)), ignore_index=True)

    # convert log value
    df["T"] = 10 ** df["logT"]
    df["P"] = 10 ** df["logP"]
    df["S"] = 10 ** df["logS"]
    df["rho"] = 10 ** df["logRho"]
    
    # convert units
    df["P"] = df["P"] * 1e9 # Gpa --> Pa
    df["rho"] = df["rho"] * 1000 #g/cm^3 --> kg/m^3
    
    # take only part of the columns
    df = df[["T", "P", "S", "rho", "grad_ad"]]
    
    return df


def read_EOS_Si(filename):
    '''
    data from https://github.com/ststewart/aneos-forsterite-2019/blob/master
    '''
    with open(filename, 'rb') as file:
        NewEOS = pickle.load(file)
    return NewEOS