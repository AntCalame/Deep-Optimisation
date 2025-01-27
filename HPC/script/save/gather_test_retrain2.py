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

parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent_parent_directory)

os.chdir(parent_parent_directory)


#--------------------------------
#---------parameters-------------
#--------------------------------

nb_transit = int(sys.argv[1])
nb_subexp = int(sys.argv[2])

#--------------------------------
#---------other-------------
#--------------------------------

nb_dim = 3
nb_tight = 3
nb_per_type = 10
nb_rep = nb_subexp
taille_pop = 1000
size_mkp = 100

parameters = {"nb_dim": 3,
              "nb_tight" : 3,
              "nb_per_type" : 10,
              "nb_rep" : nb_subexp,
              "taille_pop" : 1000,
              "size_mkp" : 100}

def read(parameters,s):
    l_dim = [5,10,30]
    
    nb_dim = parameters["nb_dim"]
    nb_tight = parameters["nb_tight"]
    nb_per_type = parameters["nb_per_type"]
    nb_rep = parameters["nb_rep"]
    size_mkp = parameters["size_mkp"]
    
    data0 = torch.load(s+"/data_test/0-0.pt")
    taille_pop = 1000
    
    Data = {"fitness" : torch.zeros((nb_dim,nb_tight,nb_per_type,nb_rep,nb_transit,nb_transit,taille_pop),dtype=torch.float),
            "time" : torch.zeros((nb_dim,nb_tight,nb_per_type,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "evaluation" : torch.zeros((nb_dim,nb_tight,nb_per_type,nb_rep,nb_transit,nb_transit),dtype=torch.float)
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
            
        for j in range(20):
            
            data = torch.load(s+"/data_test/"+str(i)+"-"+str(j)+".pt")
            
            for ss,d in data.items():
                
                Data["fitness"][idx_dim][idx_tight][idx_prb][idx_ite][j]=d["fit"]
                Data["time"][idx_dim][idx_tight][idx_prb][idx_ite][j]=d["time"]
                Data["evaluation"][idx_dim][idx_tight][idx_prb][idx_ite][j]=d["eval"]
    
    return Data

def compute(Data,parameters):
    nb_dim = parameters["nb_dim"]
    nb_tight = parameters["nb_tight"]
    nb_per_type = parameters["nb_per_type"]
    nb_rep = parameters["nb_rep"]
    taille_pop = parameters["taille_pop"]
    size_mkp = parameters["size_mkp"]
    
    save = {}
    
    save["fitness"]={}
    save["fitness"]["max"]={}
    
    # fit(-1): max 
    # mean, std, Q1, Q2, Q3, IQR, P<, P>, min, max, count1
    # global, macro, micro
    save["fitness"]["max"]["t=-1"]={}
    f_max_last = Data["fitness"][:,:,:,:,:,-1].max(-1).values
    save["fitness"]["max"]["t=-1"]["global"]= {"mean" : f_max_last.mean(),
                                               "std" : f_max_last.std(),
                                               "Q1" : torch.quantile(f_max_last.reshape(-1),0.25),
                                               "Q2" : torch.quantile(f_max_last.reshape(-1),0.5),
                                               "Q3" : torch.quantile(f_max_last.reshape(-1),0.75),
                                               "min" : f_max_last.min(),
                                               "max" : f_max_last.max(),
                                               "count1" : (f_max_last>=1).to(torch.int).mean(dtype=torch.float),
                                               "distribution": f_max_last}
    save["fitness"]["max"]["t=-1"]["global"]["IQR"]=save["fitness"]["max"]["t=-1"]["global"]["Q3"]-save["fitness"]["max"]["t=-1"]["global"]["Q1"]
    save["fitness"]["max"]["t=-1"]["global"]["sup"]=(f_max_last>save["fitness"]["max"]["t=-1"]["global"]["Q3"]+1.5*save["fitness"]["max"]["t=-1"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    save["fitness"]["max"]["t=-1"]["global"]["inf"]=(f_max_last<save["fitness"]["max"]["t=-1"]["global"]["Q1"]-1.5*save["fitness"]["max"]["t=-1"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)

    
    f_max_last_macro = f_max_last.reshape((nb_dim,nb_tight,nb_per_type*nb_rep,20))
    save["fitness"]["max"]["t=-1"]["macro"]= {"mean" : f_max_last_macro.mean(-2),
                                               "std" : f_max_last_macro.std(-2),
                                               "Q1" : torch.quantile(f_max_last_macro,0.25,dim=-2),
                                               "Q2" : torch.quantile(f_max_last_macro,0.5,dim=-2),
                                               "Q3" : torch.quantile(f_max_last_macro,0.75,dim=-2),
                                               "min" : f_max_last_macro.min(-2).values,
                                               "max" : f_max_last_macro.max(-2).values,
                                               "count1" : (f_max_last_macro>=1).to(torch.int).mean(-2,dtype=torch.float)
                                               }
    save["fitness"]["max"]["t=-1"]["macro"]["IQR"]=save["fitness"]["max"]["t=-1"]["macro"]["Q3"]-save["fitness"]["max"]["t=-1"]["macro"]["Q1"]
    save["fitness"]["max"]["t=-1"]["macro"]["sup"]=(f_max_last_macro>(save["fitness"]["max"]["t=-1"]["macro"]["Q3"]+1.5*save["fitness"]["max"]["t=-1"]["macro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    save["fitness"]["max"]["t=-1"]["macro"]["inf"]=(f_max_last_macro<(save["fitness"]["max"]["t=-1"]["macro"]["Q1"]-1.5*save["fitness"]["max"]["t=-1"]["macro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    
    save["fitness"]["max"]["t=-1"]["micro"]= {"mean" : f_max_last.mean(-2),
                                               "std" : f_max_last.std(-2),
                                               "Q1" : torch.quantile(f_max_last,0.25,dim=-2),
                                               "Q2" : torch.quantile(f_max_last,0.5,dim=-2),
                                               "Q3" : torch.quantile(f_max_last,0.75,dim=-2),
                                               "min" : f_max_last.min(-2).values,
                                               "max" : f_max_last.max(-2).values,
                                               "count1" : (f_max_last>=1).to(torch.int).mean(-2,dtype=torch.float)
                                               }
    save["fitness"]["max"]["t=-1"]["micro"]["IQR"]=save["fitness"]["max"]["t=-1"]["micro"]["Q3"]-save["fitness"]["max"]["t=-1"]["micro"]["Q1"]
    save["fitness"]["max"]["t=-1"]["micro"]["sup"]=(f_max_last>(save["fitness"]["max"]["t=-1"]["micro"]["Q3"]+1.5*save["fitness"]["max"]["t=-1"]["micro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    save["fitness"]["max"]["t=-1"]["micro"]["inf"]=(f_max_last<(save["fitness"]["max"]["t=-1"]["micro"]["Q1"]-1.5*save["fitness"]["max"]["t=-1"]["micro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)

    
    # fit(t): max
    # mean, std, count1
    # global, macro, micro
    save["fitness"]["max"]["t=all"]={}
    f_max_t = Data["fitness"].max(-1).values
    save["fitness"]["max"]["t=all"]["global"]={"mean" : f_max_t.mean((0,1,2,3)),
                                               "std" : f_max_t.std((0,1,2,3)),
                                               "count1" : (f_max_t>=1).to(torch.int).mean((0,1,2,3),dtype=torch.float)
                                               }
    save["fitness"]["max"]["t=all"]["macro"]={"mean" : f_max_t.mean((2,3)),
                                               "std" : f_max_t.std((2,3)),
                                               "count1" : (f_max_t>=1).to(torch.int).mean((2,3),dtype=torch.float)
                                               }
    save["fitness"]["max"]["t=all"]["micro"]={"mean" : f_max_t.mean(3),
                                               "std" : f_max_t.std(3),
                                               "count1" : (f_max_t>=1).to(torch.int).mean(3,dtype=torch.float)
                                               }

    save["time"]={}
    save["evaluation"]={}
    # time(-1), evaluation(-1):
    # mean, std, Q1, Q2, Q3, IQR, P<, P>, min, max
    # global, macro, micro
    save["time"]["t=-1"]={}
    save["evaluation"]["t=-1"]={}
    t_last = Data["time"][:,:,:,:,:,-1]
    save["time"]["t=-1"]["global"]= {"mean" : t_last.reshape((-1,20)).mean(0),
                                     "std" : t_last.reshape((-1,20)).std(0),
                                     "Q1" : torch.quantile(t_last.reshape((-1,20)),0.25,dim=0),
                                     "Q2" : torch.quantile(t_last.reshape((-1,20)),0.5,dim=0),
                                     "Q3" : torch.quantile(t_last.reshape((-1,20)),0.75,dim=0),
                                     "min" : t_last.reshape((-1,20)).min(0).values,
                                     "max" : t_last.reshape((-1,20)).max(0).values,
                                     "distribution": t_last
                                     }
    save["time"]["t=-1"]["global"]["IQR"]=save["time"]["t=-1"]["global"]["Q3"]-save["time"]["t=-1"]["global"]["Q1"]
    save["time"]["t=-1"]["global"]["sup"]=(t_last>save["time"]["t=-1"]["global"]["Q3"]+1.5*save["time"]["t=-1"]["global"]["IQR"]).unsqueeze(-2).to(torch.int).mean(0,dtype=torch.float)
    save["time"]["t=-1"]["global"]["inf"]=(t_last<save["time"]["t=-1"]["global"]["Q1"]-1.5*save["time"]["t=-1"]["global"]["IQR"]).unsqueeze(-2).to(torch.int).mean(0,dtype=torch.float)
    
    e_last = Data["evaluation"][:,:,:,:,:,-1]
    save["evaluation"]["t=-1"]["global"]= {"mean" : e_last.reshape((-1,20)).mean(0),
                                     "std" : e_last.reshape((-1,20)).std(0),
                                     "Q1" : torch.quantile(e_last.reshape((-1,20)),0.25,dim=0),
                                     "Q2" : torch.quantile(e_last.reshape((-1,20)),0.5,dim=0),
                                     "Q3" : torch.quantile(e_last.reshape((-1,20)),0.75,dim=0),
                                     "min" : e_last.reshape((-1,20)).min(0).values,
                                     "max" : e_last.reshape((-1,20)).max(0).values,
                                     "distribution": e_last
                                     }
    save["evaluation"]["t=-1"]["global"]["IQR"]=save["evaluation"]["t=-1"]["global"]["Q3"]-save["evaluation"]["t=-1"]["global"]["Q1"]
    save["evaluation"]["t=-1"]["global"]["sup"]=(e_last>save["evaluation"]["t=-1"]["global"]["Q3"]+1.5*save["evaluation"]["t=-1"]["global"]["IQR"]).unsqueeze(-2).to(torch.int).mean(0,dtype=torch.float)
    save["evaluation"]["t=-1"]["global"]["inf"]=(e_last<save["evaluation"]["t=-1"]["global"]["Q1"]-1.5*save["evaluation"]["t=-1"]["global"]["IQR"]).unsqueeze(-2).to(torch.int).mean(0,dtype=torch.float)
    
    t_last_macro = Data["time"][:,:,:,:,:,-1].reshape((nb_dim,nb_tight,nb_per_type*nb_rep,20))
    save["time"]["t=-1"]["macro"]= {"mean" : t_last_macro.mean(-2),
                                    "std" : t_last_macro.std(-2),
                                    "Q1" : torch.quantile(t_last_macro,0.25,-2),
                                    "Q2" : torch.quantile(t_last_macro,0.5,-2),
                                    "Q3" : torch.quantile(t_last_macro,0.75,-2),
                                    "min" : t_last_macro.min(-2).values,
                                    "max" : t_last_macro.max(-2).values
                                    }
    save["time"]["t=-1"]["macro"]["IQR"]=save["time"]["t=-1"]["macro"]["Q3"]-save["time"]["t=-1"]["macro"]["Q1"]
    save["time"]["t=-1"]["macro"]["sup"]=(t_last_macro>(save["time"]["t=-1"]["macro"]["Q3"]+1.5*save["time"]["t=-1"]["macro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    save["time"]["t=-1"]["macro"]["inf"]=(t_last_macro<(save["time"]["t=-1"]["macro"]["Q1"]-1.5*save["time"]["t=-1"]["macro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    
    e_last_macro = Data["evaluation"][:,:,:,:,:,-1].reshape((nb_dim,nb_tight,nb_per_type*nb_rep,20))
    save["evaluation"]["t=-1"]["macro"]= {"mean" : e_last_macro.mean(-2),
                                          "std" : e_last_macro.std(-2),
                                          "Q1" : torch.quantile(e_last_macro,0.25,-2),
                                          "Q2" : torch.quantile(e_last_macro,0.5,-2),
                                          "Q3" : torch.quantile(e_last_macro,0.75,-2),
                                          "min" : e_last_macro.min(-2).values,
                                          "max" : e_last_macro.max(-2).values
                                          }
    save["evaluation"]["t=-1"]["macro"]["IQR"]=save["evaluation"]["t=-1"]["macro"]["Q3"]-save["evaluation"]["t=-1"]["macro"]["Q1"]
    save["evaluation"]["t=-1"]["macro"]["sup"]=(e_last_macro>(save["evaluation"]["t=-1"]["macro"]["Q3"]+1.5*save["evaluation"]["t=-1"]["macro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    save["evaluation"]["t=-1"]["macro"]["inf"]=(e_last_macro<(save["evaluation"]["t=-1"]["macro"]["Q1"]-1.5*save["evaluation"]["t=-1"]["macro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    
    save["time"]["t=-1"]["micro"]= {"mean" : t_last.mean(-2),
                                    "std" : t_last.std(-2),
                                    "Q1" : torch.quantile(t_last,0.25,-2),
                                    "Q2" : torch.quantile(t_last,0.5,-2),
                                    "Q3" : torch.quantile(t_last,0.75,-2),
                                    "min" : t_last.min(-2).values,
                                    "max" : t_last.max(-2).values
                                    }
    save["time"]["t=-1"]["micro"]["IQR"]=save["time"]["t=-1"]["micro"]["Q3"]-save["time"]["t=-1"]["micro"]["Q1"]
    save["time"]["t=-1"]["micro"]["sup"]=(t_last>(save["time"]["t=-1"]["micro"]["Q3"]+1.5*save["time"]["t=-1"]["micro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    save["time"]["t=-1"]["micro"]["inf"]=(t_last<(save["time"]["t=-1"]["micro"]["Q1"]-1.5*save["time"]["t=-1"]["micro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    
    save["evaluation"]["t=-1"]["micro"]= {"mean" : e_last.mean(-2),
                                          "std" : e_last.std(-2),
                                          "Q1" : torch.quantile(e_last,0.25,-2),
                                          "Q2" : torch.quantile(e_last,0.5,-2),
                                          "Q3" : torch.quantile(e_last,0.75,-2),
                                          "min" : e_last.min(-2).values,
                                          "max" : e_last.max(-2).values
                                          }
    save["evaluation"]["t=-1"]["micro"]["IQR"]=save["evaluation"]["t=-1"]["micro"]["Q3"]-save["evaluation"]["t=-1"]["micro"]["Q1"]
    save["evaluation"]["t=-1"]["micro"]["sup"]=(e_last>(save["evaluation"]["t=-1"]["micro"]["Q3"]+1.5*save["evaluation"]["t=-1"]["micro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    save["evaluation"]["t=-1"]["micro"]["inf"]=(e_last<(save["evaluation"]["t=-1"]["micro"]["Q1"]-1.5*save["evaluation"]["t=-1"]["micro"]["IQR"]).unsqueeze(-2)).to(torch.int).mean(-2,dtype=torch.float)
    
    # time(t), evaluation(t):
    # mean
    # global, macro
    save["time"]["t=all"]={}
    save["evaluation"]["t=all"]={}
    
    save["time"]["t=all"] = {"global" : Data["time"].mean((0,1,2,3)),
                             "macro" : Data["time"].mean((2,3))
                             }
    save["evaluation"]["t=all"] = {"global" : Data["evaluation"].mean((0,1,2,3)),
                                   "macro" : Data["evaluation"].mean((2,3))
                                   }
    
    return save


items = os.listdir(parent_parent_directory)

Data = {}

for ii in items:
    if ii!="HPC" and ii!="sup1.pt" and ii!="data.pt" and ii!="data_test.pt" and ii!="data_retrain.pt":
        if os.path.exists(os.path.join(ii, "data_test")):
            raw_data = read(parameters,ii)
            data_processed = compute(raw_data,parameters)
            
            Data[ii] = data_processed
        
torch.save(Data, "data_retrain.pt")