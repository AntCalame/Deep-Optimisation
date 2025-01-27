import os
import sys
import torch
import numpy as np
import copy

#--------------------------------
#---------manage path------------
#--------------------------------
current_directory = os.path.dirname(os.path.realpath(__file__))

parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

path_COProblems = os.path.join(current_directory, "COProblems")
sys.path.append(path_COProblems)

path_mkp = os.path.join(path_COProblems, "mkp")
sys.path.append(path_mkp)

parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent_parent_directory)

os.chdir(parent_parent_directory)


#--------------------------------
#---------parameters-------------
#--------------------------------

from COProblems.MKP import MKP

nb_subexp = int(sys.argv[1])

nb_dim = 3
nb_tight = 3
nb_per_type = 10
nb_rep = nb_subexp
taille_pop = 1000
size_mkp = 100

parameters = {"nb_dim": 3,
              "nb_tight" : 2,
              "nb_per_type" : 10,
              "nb_rep" : nb_subexp
              }


def read(parameters,s, nb_item):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    print(device)
    device = torch.device(device)
    
    l_dim = [5,10,30]
    
    nb_dim = parameters["nb_dim"]
    nb_tight = parameters["nb_tight"]
    nb_per_type = parameters["nb_per_type"]
    nb_rep = parameters["nb_rep"]
    
    Data = {"fitness" : torch.zeros((nb_dim,nb_tight,nb_per_type,nb_rep),dtype=torch.float),
            "time" : torch.zeros((nb_dim,nb_tight,nb_per_type,nb_rep),dtype=torch.float),
            "evaluation" : torch.zeros((nb_dim,nb_tight,nb_per_type,nb_rep),dtype=torch.float)
            }
                
    nb_total = nb_dim*nb_tight*nb_per_type*nb_rep
    
    for i in range(nb_total):
        if i%100==0:
            print(i,"/",nb_total)
    
        idx_dim = int(np.floor(i/(nb_tight*nb_per_type*nb_rep)))
        dim = l_dim[idx_dim]
        idx_prb0 = int(np.floor((i-idx_dim*(nb_tight*nb_per_type*nb_rep))/(nb_rep)))
        idx_ite = int((i-idx_dim*(nb_tight*nb_per_type*nb_rep))%nb_rep)
        idx_tight = int(idx_prb0//nb_per_type)
        idx_prb = int(idx_prb0-nb_per_type*idx_tight)
        
        
        data = torch.load(s+"/data_train/"+str(i)+".pt")
        Data["fitness"][idx_dim][idx_tight][idx_prb][idx_ite] = data["best_obj_val"][-1]
        Data["time"][idx_dim][idx_tight][idx_prb][idx_ite] = data["time"][-1]
        Data["evaluation"][idx_dim][idx_tight][idx_prb][idx_ite] = data["evaluations"][-1]
                            
    return Data    

def compute(Data,parameters):
    nb_dim = parameters["nb_dim"]
    nb_tight = parameters["nb_tight"]
    nb_per_type = parameters["nb_per_type"]
    nb_rep = parameters["nb_rep"]
    save = {}
    
    save["fitness"]={}
    save["fitness"]["max"]={}
    # fit(-1): max 
    # mean, std, Q1, Q2, Q3, IQR, P<, P>, min, max, count1
    # global, macro, micro
    save["fitness"]["max"]={}
    f_max_last = Data["fitness"]
    save["fitness"]["max"]["global"]= {"mean" : f_max_last.mean(),
                                               "std" : f_max_last.std(),
                                               "Q1" : torch.quantile(f_max_last.reshape(-1),0.25),
                                               "Q2" : torch.quantile(f_max_last.reshape(-1),0.5),
                                               "Q3" : torch.quantile(f_max_last.reshape(-1),0.75),
                                               "min" : f_max_last.min(),
                                               "max" : f_max_last.max(),
                                               "count1" : (f_max_last>=1).to(torch.int).mean(dtype=torch.float),
                                               "distribution": f_max_last}
    save["fitness"]["max"]["global"]["IQR"]=save["fitness"]["max"]["global"]["Q3"]-save["fitness"]["max"]["global"]["Q1"]
    save["fitness"]["max"]["global"]["sup"]=(f_max_last>save["fitness"]["max"]["global"]["Q3"]+1.5*save["fitness"]["max"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    save["fitness"]["max"]["global"]["inf"]=(f_max_last<save["fitness"]["max"]["global"]["Q1"]-1.5*save["fitness"]["max"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)

    
    f_max_last_macro = f_max_last.reshape((nb_dim,nb_tight,nb_per_type*nb_rep))
    save["fitness"]["max"]["macro"]= {"mean" : f_max_last_macro.mean(-1),
                                               "std" : f_max_last_macro.std(-1),
                                               "Q1" : torch.quantile(f_max_last_macro,0.25,dim=-1),
                                               "Q2" : torch.quantile(f_max_last_macro,0.5,dim=-1),
                                               "Q3" : torch.quantile(f_max_last_macro,0.75,dim=-1),
                                               "min" : f_max_last_macro.min(-1).values,
                                               "max" : f_max_last_macro.max(-1).values,
                                               "count1" : (f_max_last_macro>=1).to(torch.int).mean(-1,dtype=torch.float)
                                               }
    save["fitness"]["max"]["macro"]["IQR"]=save["fitness"]["max"]["macro"]["Q3"]-save["fitness"]["max"]["macro"]["Q1"]
    save["fitness"]["max"]["macro"]["sup"]=(f_max_last_macro>(save["fitness"]["max"]["macro"]["Q3"]+1.5*save["fitness"]["max"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["fitness"]["max"]["macro"]["inf"]=(f_max_last_macro<(save["fitness"]["max"]["macro"]["Q1"]-1.5*save["fitness"]["max"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    save["fitness"]["max"]["micro"]= {"mean" : f_max_last.mean(-1),
                                               "std" : f_max_last.std(-1),
                                               "Q1" : torch.quantile(f_max_last,0.25,dim=-1),
                                               "Q2" : torch.quantile(f_max_last,0.5,dim=-1),
                                               "Q3" : torch.quantile(f_max_last,0.75,dim=-1),
                                               "min" : f_max_last.min(-1).values,
                                               "max" : f_max_last.max(-1).values,
                                               "count1" : (f_max_last>=1).to(torch.int).mean(-1,dtype=torch.float)
                                               }
    save["fitness"]["max"]["micro"]["IQR"]=save["fitness"]["max"]["micro"]["Q3"]-save["fitness"]["max"]["micro"]["Q1"]
    save["fitness"]["max"]["micro"]["sup"]=(f_max_last>(save["fitness"]["max"]["micro"]["Q3"]+1.5*save["fitness"]["max"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["fitness"]["max"]["micro"]["inf"]=(f_max_last<(save["fitness"]["max"]["micro"]["Q1"]-1.5*save["fitness"]["max"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    save["time"]={}
    save["evaluation"]={}
    # time(-1), evaluation(-1):
    # mean, std, Q1, Q2, Q3, IQR, P<, P>, min, max
    # global, macro, micro
    save["time"]={}
    save["evaluation"]={}
    t_last = Data["time"]
    save["time"]["global"]= {"mean" : t_last.mean(),
                                     "std" : t_last.std(),
                                     "Q1" : torch.quantile(t_last.reshape(-1),0.25),
                                     "Q2" : torch.quantile(t_last.reshape(-1),0.5),
                                     "Q3" : torch.quantile(t_last.reshape(-1),0.75),
                                     "min" : t_last.min(),
                                     "max" : t_last.max(),
                                     "distribution": t_last
                                     }
    save["time"]["global"]["IQR"]=save["time"]["global"]["Q3"]-save["time"]["global"]["Q1"]
    save["time"]["global"]["sup"]=(t_last>save["time"]["global"]["Q3"]+1.5*save["time"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    save["time"]["global"]["inf"]=(t_last<save["time"]["global"]["Q1"]-1.5*save["time"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    
    e_last = Data["evaluation"]
    save["evaluation"]["global"]= {"mean" : e_last.mean(),
                                     "std" : e_last.std(),
                                     "Q1" : torch.quantile(e_last.reshape(-1),0.25),
                                     "Q2" : torch.quantile(e_last.reshape(-1),0.5),
                                     "Q3" : torch.quantile(e_last.reshape(-1),0.75),
                                     "min" : e_last.min(),
                                     "max" : e_last.max(),
                                     "distribution": e_last
                                     }
    save["evaluation"]["global"]["IQR"]=save["evaluation"]["global"]["Q3"]-save["evaluation"]["global"]["Q1"]
    save["evaluation"]["global"]["sup"]=(e_last>save["evaluation"]["global"]["Q3"]+1.5*save["evaluation"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    save["evaluation"]["global"]["inf"]=(e_last<save["evaluation"]["global"]["Q1"]-1.5*save["evaluation"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    
    t_last_macro = Data["time"].reshape((nb_dim,nb_tight,nb_per_type*nb_rep))
    save["time"]["macro"]= {"mean" : t_last_macro.mean(-1),
                                    "std" : t_last_macro.std(-1),
                                    "Q1" : torch.quantile(t_last_macro,0.25,-1),
                                    "Q2" : torch.quantile(t_last_macro,0.5,-1),
                                    "Q3" : torch.quantile(t_last_macro,0.75,-1),
                                    "min" : t_last_macro.min(-1).values,
                                    "max" : t_last_macro.max(-1).values
                                    }
    save["time"]["macro"]["IQR"]=save["time"]["macro"]["Q3"]-save["time"]["macro"]["Q1"]
    save["time"]["macro"]["sup"]=(t_last_macro>(save["time"]["macro"]["Q3"]+1.5*save["time"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["time"]["macro"]["inf"]=(t_last_macro<(save["time"]["macro"]["Q1"]-1.5*save["time"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    e_last_macro = Data["evaluation"].reshape((nb_dim,nb_tight,nb_per_type*nb_rep))
    save["evaluation"]["macro"]= {"mean" : e_last_macro.mean(-1),
                                          "std" : e_last_macro.std(-1),
                                          "Q1" : torch.quantile(e_last_macro,0.25,-1),
                                          "Q2" : torch.quantile(e_last_macro,0.5,-1),
                                          "Q3" : torch.quantile(e_last_macro,0.75,-1),
                                          "min" : e_last_macro.min(-1).values,
                                          "max" : e_last_macro.max(-1).values
                                          }
    save["evaluation"]["macro"]["IQR"]=save["evaluation"]["macro"]["Q3"]-save["evaluation"]["macro"]["Q1"]
    save["evaluation"]["macro"]["sup"]=(e_last_macro>(save["evaluation"]["macro"]["Q3"]+1.5*save["evaluation"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["evaluation"]["macro"]["inf"]=(e_last_macro<(save["evaluation"]["macro"]["Q1"]-1.5*save["evaluation"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    save["time"]["micro"]= {"mean" : t_last.mean(-1),
                                    "std" : t_last.std(-1),
                                    "Q1" : torch.quantile(t_last,0.25,-1),
                                    "Q2" : torch.quantile(t_last,0.5,-1),
                                    "Q3" : torch.quantile(t_last,0.75,-1),
                                    "min" : t_last.min(-1).values,
                                    "max" : t_last.max(-1).values
                                    }
    save["time"]["micro"]["IQR"]=save["time"]["micro"]["Q3"]-save["time"]["micro"]["Q1"]
    save["time"]["micro"]["sup"]=(t_last>(save["time"]["micro"]["Q3"]+1.5*save["time"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["time"]["micro"]["inf"]=(t_last<(save["time"]["micro"]["Q1"]-1.5*save["time"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    save["evaluation"]["micro"]= {"mean" : e_last.mean(-1),
                                          "std" : e_last.std(-1),
                                          "Q1" : torch.quantile(e_last,0.25,-1),
                                          "Q2" : torch.quantile(e_last,0.5,-1),
                                          "Q3" : torch.quantile(e_last,0.75,-1),
                                          "min" : e_last.min(-1).values,
                                          "max" : e_last.max(-1).values
                                          }
    save["evaluation"]["micro"]["IQR"]=save["evaluation"]["micro"]["Q3"]-save["evaluation"]["micro"]["Q1"]
    save["evaluation"]["micro"]["sup"]=(e_last>(save["evaluation"]["micro"]["Q3"]+1.5*save["evaluation"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["evaluation"]["micro"]["inf"]=(e_last<(save["evaluation"]["micro"]["Q1"]-1.5*save["evaluation"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    return save
    
items = os.listdir(parent_parent_directory)

Data = {}

for ii in items:
    if ii!="HPC" and ii!="sup1.pt" and ii!="data.pt" and ii!="data_test.pt":
        nb_item = ii[:3]
        
        raw_data = read(parameters,ii,nb_item)
        data_processed = compute(raw_data,parameters)
        
        Data[ii] = data_processed
        
torch.save(Data, "data.pt")