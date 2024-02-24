
from gamspy import (
    Container,
    Set,
    Parameter,
    Variable,
    Equation,
    Model,
    Sum,
    Ord,
)
import pandas as pd
import os
import sys
from gamspy.math import exp


# HELPER FUNCTIONS:-------------------------------------------------------
def filter_data(price_table, node):

    nodal_price = price_table.loc[
        price_table.node == node, ["t", "k", "marginal"]
    ]
    df_sorted = nodal_price.sort_values(by=["k", "t", "marginal"])
    df_sorted = df_sorted.drop(["k", "t"], axis=1)
    df_sorted.reset_index(inplace=True, drop=True)
    df_sorted["hour"] = df_sorted.index
    return df_sorted


def filter_gridload(df_load, node):

    df_load = df_load.loc[df_load.node == node, ["t", "k", "value"]]
    df_sorted = df_load.sort_values(by=["k", "t", "value"])
    df_sorted = df_sorted.drop(["k", "t"], axis=1)
    df_sorted.reset_index(inplace=True, drop=True)
    return df_sorted


def main():
    
    
    hselect = 100 # less than or equal 8760
    d = 1 / 4
    size = 100

    price = pd.read_csv("sample_price.csv")
    load_df = pd.read_csv("sample_load.csv")
    # Filter the data
    df = filter_data(price, "dena21")
    nodal_load = filter_gridload(load_df, "dena21").value

    df=df.head(hselect)
    nodal_load = nodal_load.head(hselect)

    # MODEL INITIALIZATION 
    bat = Container(working_directory=os.path.join(os.getcwd(), "debugg_bat"))

    # DEFINIE SETS  and Parameters
    t = Set(bat, name="t", records=df.hour.tolist(), description=" hours")
    P = Parameter(
        bat,
        "P",
        domain=[t],
        records=df["marginal"],
        description="marginal price",
    )

    # Variables
    SOC = Variable(bat, name="SOC", type="free", domain=t)
    Pd = Variable(bat, name="Pd", type="Positive", domain=t)
    Pc = Variable(bat, name="Pc", type="Positive", domain=t)

    obj = Variable(
        bat, name="obj", type="free", description="Objective function"
    )

    # SCALARS
    SOC0 = Parameter(bat, name="SOC0", records=size / 2)
    SOCmax = Parameter(bat, name="SOCmax", records=size)
    eta_c = Parameter(bat, name="eta_c", records=0.95)
    eta_d = Parameter(bat, name="eta_d", records=0.95)

    SOC.up[t] = SOCmax
    SOC.lo[t] = d * SOCmax

    Pc.up[t] = d * SOCmax
    Pc.lo[t] = 0
    Pd.up[t] = d * SOCmax
    Pd.lo[t] = 0

    # EQUATION
    constESS = Equation(bat, name="constESS", type="regular", domain=t)

    defobj = Equation(bat, "defobj")
    
    constESS[t] = (
        SOC[t]
        == SOC0.where[Ord(t) == 1]
        + SOC[t.lag(1)].where[Ord(t) > 1]
        + Pc[t] * eta_c
        - Pd[t] / eta_d 
    )

    
    
    level = Variable(
        bat, "level", domain=[t], type="Positive"
    )
    
    # NL PIECEWISE FUNCTION
    base = 0
    high = 10e3
    
    x1 = -5e3
    x2 = -1e3
    x3 = 1e3
    x4 = 5e3
    
    load = Parameter(bat, 'load', domain=[t], records= nodal_load)
    net_l = Variable(bat, 'net_l', domain=[t], type='free')
    defnetl = Equation(bat, name="defnetl", domain=[t])
    
    deflevel = Equation(bat, name="deflevel", domain=[t])
    defnetl [t] = net_l[t] == load[t] + Pc[t] - Pd[t]
    
    # Define piecewise function
    def sigmoid(x, x0, k):
        return 1 / (1 + exp(-k * (x - x0)))

    def _function(x, x1, x2, x3, x4, high, base):
        k1 = (high + base) / (x2 - x1)
        k3 = (high - base) / (x4 - x3)

        term1 = k1 * (x - x1) - high
        term2 = base
        term3 = k3 * (x - x3) + base
        
        n=10

        return (term1 * sigmoid(x, x1, n) * sigmoid(x2, x, n)
                + term2 * sigmoid(x, x2, n) * sigmoid(x3, x, n)
                + term3 * sigmoid(x, x3, n) * sigmoid(x4, x, n))

    deflevel[t] = level[t] == _function(net_l[t], x1, x2, x3, x4, high, base)
    
        
    # define ojective function
    defobj[...] = obj == Sum(t, (Pd[t] - Pc[t]) * (P[t] + level[t])) 

    m = Model(
    bat,
    name="m",
    equations=bat.getEquations(),
    problem="NLP", 
    sense="MAX",
    objective=obj,)
    
    m.solve(output=sys.stdout, solver="SCIP", )
  
if __name__ == "__main__":
    main()
    




