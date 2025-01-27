import torch

def forward_specific(x, model, layer):
    z = model.encoder[:1+(2*layer)](x)
    reconstructed = model.decoder[-2*layer-1:](z)
    return reconstructed

def analyse(store,
            max_fitness,
            store_model,
            nb_iteration,
            max_depth):
    
    save = {"fitness" : torch.cat(store["fitness"],0)/max_fitness,
            "time" : torch.FloatTensor(store["time"]),
            "evaluation" : torch.FloatTensor(store["evaluation"])}

    save_population = torch.cat(store["population"],0)
    
    # compute loss per population per model
    pred = []
    for i,model_i in enumerate(store_model):
        model_i.eval()
        l_rec = []
        for j in range(nb_iteration):
            rec_j = model_i.forward(store["population"][j][0])[0]
            l_rec += [rec_j[None,...]]
        pred += [torch.cat(l_rec)[None,...]]
    reconstructed = torch.cat(pred)

    loss = ((save_population[None,...]-reconstructed)**2).mean((-2,-1)).detach()
    save["loss"] = loss
    
    # compute correlation between fitness and loss
    individual_loss = ((save_population[None,...]-reconstructed)**2).mean(-1)
    covariance = ((individual_loss-individual_loss.mean(-1,keepdim=True))
                  *(save["fitness"]-save["fitness"].mean(-1,keepdim=True))[None,...]).mean(-1)
    correlation = covariance/(torch.std(save["fitness"],1)[None,...]+1e-9)/(torch.std(individual_loss,-1)+1e-9)
    save["correlation"] = correlation.detach()
    save["covariance"] = covariance.detach()
    save["c_fitness_std"] = torch.std(save["fitness"],1)[None,...].detach()+1e-9
    save["c_loss_std"] = torch.std(individual_loss,-1).detach()+1e-9
    
    
    # compute loss per population per model
    # adapted = population at t, reconstructed using layers available at t
    pred = []
    for i,model_i in enumerate(store_model):
        model_i.eval()
        l_rec = []
        for j in range(nb_iteration):
            layer = min(1+j,max_depth)
            rec_j = forward_specific(store["population"][j][0], model_i, layer)
            l_rec += [rec_j[None,...]]
        pred += [torch.cat(l_rec)[None,...]]
    reconstructed = torch.cat(pred)
    
    loss = ((save_population[None,...]-reconstructed)**2).mean((-2,-1)).detach()
    save["loss_adapted"] = loss
    
    # store individuals if fitness better than 1
    indices_sup1 = save["fitness"]>1
    sol_sup1 = save_population[indices_sup1]
    save["sup1"] = [save["fitness"][indices_sup1],
                    sol_sup1]
    
    # heatmap of first layer parameter
    model_encoder = [i.state_dict()["encoder.1.weight"][None,None,...] for i in store_model]
    distance_layer_encoder = torch.linalg.vector_norm(torch.cat(model_encoder,0)-torch.cat(model_encoder,1),2,(-2,-1))
    distance_layer_encoder_rescaled = (torch.linalg.vector_norm(torch.cat(model_encoder,0)-torch.cat(model_encoder,1),2,(-2,-1))
                                       /(torch.linalg.vector_norm(torch.cat(model_encoder,0),2,(-2,-1))+1e-9)
                                       /(torch.linalg.vector_norm(torch.cat(model_encoder,1),2,(-2,-1))+1e-9))
    
    save["distance_layer1"] = distance_layer_encoder
    save["distance_layer1_rescaled"] = distance_layer_encoder_rescaled
    
    transition_loss = {}
    for i in store["transition_loss"][0].keys():
        partial_loss = []
        for j in store["transition_loss"]:
            partial_loss += [j[i][None,...]]
        transition_loss[i] = torch.cat(partial_loss,0).detach()
    save["transition_loss"]=transition_loss

    return save