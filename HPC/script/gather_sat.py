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

parameters = {"nb_problem" : 38,
              "nb_rep" : nb_subexp,
              "taille_pop" : 1000,
              "size_mkp" : 100}

def read(parameters,s):
    
    nb_problem = parameters["nb_problem"]
    nb_rep = parameters["nb_rep"]
    size_mkp = parameters["size_mkp"]
    
    data0 = torch.load(s+"/data_train/0.pt")
    taille_pop = data0["fitness"].size(1)
    
    Data = {"fitness" : torch.zeros((nb_problem,nb_rep,nb_transit,taille_pop),dtype=torch.float),
            "loss" : torch.zeros((nb_problem,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "loss_adapted" : torch.zeros((nb_problem,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "correlation" : torch.zeros((nb_problem,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "covariance" : torch.zeros((nb_problem,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "c_fitness_std" : torch.zeros((nb_problem,nb_rep,1,nb_transit),dtype=torch.float),
            "c_loss_std" : torch.zeros((nb_problem,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "dl1" : torch.zeros((nb_problem,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "dl1_r" : torch.zeros((nb_problem,nb_rep,nb_transit,nb_transit),dtype=torch.float),
            "time" : torch.zeros((nb_problem,nb_rep,nb_transit),dtype=torch.float),
            "evaluation" : torch.zeros((nb_problem,nb_rep,nb_transit),dtype=torch.float)
            }
                
    nb_total = nb_problem*nb_rep
    for i in range(nb_total):
        
        if i%100==0:
            print(i,"/",nb_total)
    
        idx_prb = int(i//10)
        idx_ite = int(i%10)
        
        data = torch.load(s+"/data_train/"+str(i)+".pt")
        Data["fitness"][idx_prb][idx_ite] = data["fitness"].detach()
        Data["loss"][idx_prb][idx_ite] = data["loss"].detach()
        Data["loss_adapted"][idx_prb][idx_ite] = data["loss_adapted"].detach()
        Data["correlation"][idx_prb][idx_ite] = data["correlation"].detach()
        Data["covariance"][idx_prb][idx_ite] = data["covariance"].detach()
        Data["c_fitness_std"][idx_prb][idx_ite] = data["c_fitness_std"].detach()
        Data["c_loss_std"][idx_prb][idx_ite] = data["c_loss_std"].detach()
        Data["dl1"][idx_prb][idx_ite] = data["distance_layer1"].detach()
        Data["dl1_r"][idx_prb][idx_ite] = data["distance_layer1_rescaled"].detach()
        Data["time"][idx_prb][idx_ite] = data["time"].detach()
        Data["evaluation"][idx_prb][idx_ite] = data["evaluation"].detach()
        
        for i in data["transition_loss"].keys():
            if "tl_"+i not in Data.keys():
                nb_point = data["transition_loss"][i].size(-1)
                Data["tl_"+i]=torch.zeros((nb_problem,nb_rep,nb_transit,nb_point),dtype=torch.float)
                Data["tl_"+i][idx_prb][idx_ite]=data["transition_loss"][i]
            else:
                Data["tl_"+i][idx_prb][idx_ite]=data["transition_loss"][i]
                
    return Data    

def compute(Data,parameters):
    nb_problem = parameters["nb_problem"]
    nb_rep = parameters["nb_rep"]
    size_mkp = parameters["size_mkp"]
    
    save = {}
    
    save["fitness"]={}
    save["fitness"]["max"]={}
    # fit(-1): max 
    # mean, std, Q1, Q2, Q3, IQR, P<, P>, min, max, count1
    # global, macro, micro
    save["fitness"]["max"]["t=-1"]={}
    f_max_last = Data["fitness"][:,:,-1].max(-1).values
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
    
    save["fitness"]["max"]["t=-1"]["micro"]= {"mean" : f_max_last.mean(-1),
                                               "std" : f_max_last.std(-1),
                                               "Q1" : torch.quantile(f_max_last,0.25,dim=-1),
                                               "Q2" : torch.quantile(f_max_last,0.5,dim=-1),
                                               "Q3" : torch.quantile(f_max_last,0.75,dim=-1),
                                               "min" : f_max_last.min(-1).values,
                                               "max" : f_max_last.max(-1).values,
                                               "count1" : (f_max_last>=1).to(torch.int).mean(-1,dtype=torch.float)
                                               }
    save["fitness"]["max"]["t=-1"]["micro"]["IQR"]=save["fitness"]["max"]["t=-1"]["micro"]["Q3"]-save["fitness"]["max"]["t=-1"]["micro"]["Q1"]
    save["fitness"]["max"]["t=-1"]["micro"]["sup"]=(f_max_last>(save["fitness"]["max"]["t=-1"]["micro"]["Q3"]+1.5*save["fitness"]["max"]["t=-1"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["fitness"]["max"]["t=-1"]["micro"]["inf"]=(f_max_last<(save["fitness"]["max"]["t=-1"]["micro"]["Q1"]-1.5*save["fitness"]["max"]["t=-1"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    # fit(t): max
    # mean, std, count1
    # global, macro, micro
    save["fitness"]["max"]["t=all"]={}
    f_max_t = Data["fitness"].max(-1).values
    save["fitness"]["max"]["t=all"]["global"]={"mean" : f_max_t.mean((0,1)),
                                               "std" : f_max_t.std((0,1)),
                                               "count1" : (f_max_t>=1).to(torch.int).mean((0,1),dtype=torch.float)
                                               }
    save["fitness"]["max"]["t=all"]["micro"]={"mean" : f_max_t.mean(1),
                                               "std" : f_max_t.std(1),
                                               "count1" : (f_max_t>=1).to(torch.int).mean(1,dtype=torch.float)
                                               }
    
    # fit(t): mean
    # mean, std
    # global, macro, micro
    f_mean_t = Data["fitness"].mean(-1)
    save["fitness"]["mean"]={}
    save["fitness"]["mean"]["global"]={"mean" : f_mean_t.mean((0,1)),
                                       "std" : f_mean_t.std((0,1))
                                       }

    save["fitness"]["mean"]["micro"]={"mean" : f_mean_t.mean(-1),
                                       "std" : f_mean_t.std(-1)
                                       }
    save["time"]={}
    save["evaluation"]={}
    # time(-1), evaluation(-1):
    # mean, std, Q1, Q2, Q3, IQR, P<, P>, min, max
    # global, macro, micro
    save["time"]["t=-1"]={}
    save["evaluation"]["t=-1"]={}
    t_last = Data["time"][:,:,-1]
    save["time"]["t=-1"]["global"]= {"mean" : t_last.mean(),
                                     "std" : t_last.std(),
                                     "Q1" : torch.quantile(t_last.reshape(-1),0.25),
                                     "Q2" : torch.quantile(t_last.reshape(-1),0.5),
                                     "Q3" : torch.quantile(t_last.reshape(-1),0.75),
                                     "min" : t_last.min(),
                                     "max" : t_last.max(),
                                     "distribution": t_last
                                     }
    save["time"]["t=-1"]["global"]["IQR"]=save["time"]["t=-1"]["global"]["Q3"]-save["time"]["t=-1"]["global"]["Q1"]
    save["time"]["t=-1"]["global"]["sup"]=(t_last>save["time"]["t=-1"]["global"]["Q3"]+1.5*save["time"]["t=-1"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    save["time"]["t=-1"]["global"]["inf"]=(t_last<save["time"]["t=-1"]["global"]["Q1"]-1.5*save["time"]["t=-1"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    
    e_last = Data["evaluation"][:,:,-1]
    save["evaluation"]["t=-1"]["global"]= {"mean" : e_last.mean(),
                                     "std" : e_last.std(),
                                     "Q1" : torch.quantile(e_last.reshape(-1),0.25),
                                     "Q2" : torch.quantile(e_last.reshape(-1),0.5),
                                     "Q3" : torch.quantile(e_last.reshape(-1),0.75),
                                     "min" : e_last.min(),
                                     "max" : e_last.max(),
                                     "distribution": e_last
                                     }
    save["evaluation"]["t=-1"]["global"]["IQR"]=save["evaluation"]["t=-1"]["global"]["Q3"]-save["evaluation"]["t=-1"]["global"]["Q1"]
    save["evaluation"]["t=-1"]["global"]["sup"]=(e_last>save["evaluation"]["t=-1"]["global"]["Q3"]+1.5*save["evaluation"]["t=-1"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    save["evaluation"]["t=-1"]["global"]["inf"]=(e_last<save["evaluation"]["t=-1"]["global"]["Q1"]-1.5*save["evaluation"]["t=-1"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    
    save["time"]["t=-1"]["micro"]= {"mean" : t_last.mean(-1),
                                    "std" : t_last.std(-1),
                                    "Q1" : torch.quantile(t_last,0.25,-1),
                                    "Q2" : torch.quantile(t_last,0.5,-1),
                                    "Q3" : torch.quantile(t_last,0.75,-1),
                                    "min" : t_last.min(-1).values,
                                    "max" : t_last.max(-1).values
                                    }
    save["time"]["t=-1"]["micro"]["IQR"]=save["time"]["t=-1"]["micro"]["Q3"]-save["time"]["t=-1"]["micro"]["Q1"]
    save["time"]["t=-1"]["micro"]["sup"]=(t_last>(save["time"]["t=-1"]["micro"]["Q3"]+1.5*save["time"]["t=-1"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["time"]["t=-1"]["micro"]["inf"]=(t_last<(save["time"]["t=-1"]["micro"]["Q1"]-1.5*save["time"]["t=-1"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    save["evaluation"]["t=-1"]["micro"]= {"mean" : e_last.mean(-1),
                                          "std" : e_last.std(-1),
                                          "Q1" : torch.quantile(e_last,0.25,-1),
                                          "Q2" : torch.quantile(e_last,0.5,-1),
                                          "Q3" : torch.quantile(e_last,0.75,-1),
                                          "min" : e_last.min(-1).values,
                                          "max" : e_last.max(-1).values
                                          }
    save["evaluation"]["t=-1"]["micro"]["IQR"]=save["evaluation"]["t=-1"]["micro"]["Q3"]-save["evaluation"]["t=-1"]["micro"]["Q1"]
    save["evaluation"]["t=-1"]["micro"]["sup"]=(e_last>(save["evaluation"]["t=-1"]["micro"]["Q3"]+1.5*save["evaluation"]["t=-1"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["evaluation"]["t=-1"]["micro"]["inf"]=(e_last<(save["evaluation"]["t=-1"]["micro"]["Q1"]-1.5*save["evaluation"]["t=-1"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    # time(t), evaluation(t):
    # mean
    # global, macro
    save["time"]["t=all"]={}
    save["evaluation"]["t=all"]={}
    
    save["time"]["t=all"] = {"global" : Data["time"].mean((0,1))
                             }
    save["evaluation"]["t=all"] = {"global" : Data["evaluation"].mean((0,1))
                                   }
    
    # evaluations to solution:
    # mean, std, Q1, Q2, Q3, IQR, P<, P>
    # global, macro, micro
    """
    save["eval_to_sol"]={}
    cond = (Data["fitness"].max(-1).values>=1)
    cond = torch.cumsum(cond.to(torch.int),-1)
    cond = torch.where(cond==0, Data["fitness"].size(-2), cond)
    cond = torch.min(cond,-1)
    not_reach = (cond.values==Data["fitness"].size(-2))
    cond_corrected = torch.where(not_reach,
                                 Data["fitness"].size(-2)-1,cond.indices)
    eval_to_sol = torch.squeeze(torch.gather(Data["evaluation"],4,cond_corrected[...,None]))
    save["eval_to_sol"]["global"]= {"mean" : eval_to_sol.mean(),
                                    "std" : eval_to_sol.std(),
                                    "Q1" : torch.quantile(eval_to_sol.reshape(-1),0.25),
                                    "Q2" : torch.quantile(eval_to_sol.reshape(-1),0.5),
                                    "Q3" : torch.quantile(eval_to_sol.reshape(-1),0.75),
                                    "min" : eval_to_sol.min(),
                                    "max" : eval_to_sol.max()
                                    }
    save["eval_to_sol"]["global"]["IQR"]=save["eval_to_sol"]["global"]["Q3"]-save["eval_to_sol"]["global"]["Q1"]
    save["eval_to_sol"]["global"]["sup"]=(eval_to_sol>save["eval_to_sol"]["global"]["Q3"]+1.5*save["eval_to_sol"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)
    save["eval_to_sol"]["global"]["inf"]=(eval_to_sol<save["eval_to_sol"]["global"]["Q1"]-1.5*save["eval_to_sol"]["global"]["IQR"]).to(torch.int).mean(dtype=torch.float)

    
    eval_to_sol_macro = eval_to_sol.reshape((nb_problem*nb_rep))
    save["eval_to_sol"]["macro"]= {"mean" : eval_to_sol_macro.mean(-1),
                                   "std" : eval_to_sol_macro.std(-1),
                                   "Q1" : torch.quantile(eval_to_sol_macro,0.25,dim=-1),
                                   "Q2" : torch.quantile(eval_to_sol_macro,0.5,dim=-1),
                                   "Q3" : torch.quantile(eval_to_sol_macro,0.75,dim=-1),
                                   "min" : eval_to_sol_macro.min(-1).values,
                                   "max" : eval_to_sol_macro.max(-1).values,
                                   }
    save["eval_to_sol"]["macro"]["IQR"]=save["eval_to_sol"]["macro"]["Q3"]-save["eval_to_sol"]["macro"]["Q1"]
    save["eval_to_sol"]["macro"]["sup"]=(eval_to_sol_macro>(save["eval_to_sol"]["macro"]["Q3"]+1.5*save["eval_to_sol"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["eval_to_sol"]["macro"]["inf"]=(eval_to_sol_macro<(save["eval_to_sol"]["macro"]["Q1"]-1.5*save["eval_to_sol"]["macro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    
    save["eval_to_sol"]["micro"]= {"mean" : eval_to_sol.mean(-1),
                                   "std" : eval_to_sol.std(-1),
                                   "Q1" : torch.quantile(eval_to_sol,0.25,dim=-1),
                                   "Q2" : torch.quantile(eval_to_sol,0.5,dim=-1),
                                   "Q3" : torch.quantile(eval_to_sol,0.75,dim=-1),
                                   "min" : eval_to_sol.min(-1).values,
                                   "max" : eval_to_sol.max(-1).values,
                                   }
    save["eval_to_sol"]["micro"]["IQR"]=save["eval_to_sol"]["micro"]["Q3"]-save["eval_to_sol"]["micro"]["Q1"]
    save["eval_to_sol"]["micro"]["sup"]=(eval_to_sol>(save["eval_to_sol"]["micro"]["Q3"]+1.5*save["eval_to_sol"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    save["eval_to_sol"]["micro"]["inf"]=(eval_to_sol<(save["eval_to_sol"]["micro"]["Q1"]-1.5*save["eval_to_sol"]["micro"]["IQR"])[...,None]).to(torch.int).mean(-1,dtype=torch.float)
    """
    # loss
    # mean, std
    # global, macro    
    save["loss"]={"global" : {"mean" : Data["loss"].mean((0,1)),
                              "std" : Data["loss"].std((0,1))}
                  }
    
    # loss_adapted
    # mean, std
    # global, macro  
    save["loss_adapted"]={"global" : {"mean" : Data["loss_adapted"].mean((0,1)),
                                      "std" : Data["loss_adapted"].std((0,1))}
                          }
    
    # covariance
    # mean, std
    # global, macro
    save["covariance"]={"global" : {"mean" : Data["covariance"].mean((0,1)),
                                    "std" : Data["covariance"].std((0,1))}
                        }
    
    # correlation
    # mean, std
    # global, macro
    corr = (Data["correlation"].abs()<1).to(torch.int)*Data["correlation"]
    save["correlation"]={"global" : {"mean" : corr.mean((0,1)),
                                     "std" : corr.std((0,1))}
                         }
    
    # std fitness
    # mean, std
    # global, macro  
    save["c_fitness_std"]={"global" : {"mean" : Data["c_fitness_std"].mean((0,1)),
                                       "std" : Data["c_fitness_std"].std((0,1))}
                           }
    
    # std loss
    # mean, std
    # global, macro  
    save["c_loss_std"]={"global" : {"mean" : Data["c_loss_std"].mean((0,1)),
                                    "std" : Data["c_loss_std"].std((0,1))}
                        }
    
    # distance layer1
    # mean, std
    # global, macro  
    save["dl1"]={"global" : {"mean" : Data["dl1"].mean((0,1)),
                             "std" : Data["dl1"].std((0,1))}
                 }
    
    # all components of loss:
    # mean
    # global, macro, micro
    save["transition_loss"]={}
    for i in Data.keys():
        if i[:3]=="tl_":
            save["transition_loss"][i[3:]]={"global" : Data[i].mean((0,1)),
                                            "micro" : Data[i].mean((1))
                                            }
    
    return save

items = os.listdir(parent_parent_directory)

#dsup_1 = {}
Data = {}

for ii in items:
    if ii!="HPC" and ii!="sup1.pt" and ii!="data.pt" and ii!="data_test.pt":
        #raw_data, d_sup1 = read(parameters,ii)
        raw_data = read(parameters,ii)
        data_processed = compute(raw_data,parameters)
        
        Data[ii] = data_processed
        #dsup_1[ii] = d_sup1
        
torch.save(Data, "data.pt")
#torch.save(dsup_1,"sup1.pt")
