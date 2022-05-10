import torch
import pandas as pd

def add_oui_and_oiu(graph):
    edges = graph[('u', 'b', 'i')].edge_index

    df = pd.DataFrame({'u':edges[0], 'i':edges[1]})

    # KLOPT NIKS VAN
    # sort in same way as pytorch will do
    df = df.sort_values(['u', 'i'])

    oui = df.groupby("u")['i'].rank("first")
    oiu = df.groupby("i")['u'].rank("first")
    
    graph[('u', 'b', 'i')].oui = oui.values
    graph[('u', 'b', 'i')].oiu = oiu.values

