import torch

# 65 different files
partial_list_of_file = ['1', '4', '5', '6', '7', '8', '9', '10',
 '11', '12', '13', '14', '15', '16', '17', '18', '19',
 '201', '202', '203', '205', '207', '208', '209', '210',
 '211', '212', '214', '215', '216', '217', '218', '219', '220',
 '301', '302', '303', '304', '305', '306', '307', '308', '309', '310']

current_list_of_file = ['1', '10',
 '11', '12', '13', '14', '15', '16', '17', '18', '19',
 '201', '202', '203', '205', '207', '208', '209', '210',
 '211', '212', '214', '215', '216', '217', '218', '219', '220',
 '301', '302', '303', '304', '305', '306', '307', '308', '309', '310']

def SATpopulate(idx : int) -> tuple[torch.Tensor, torch.Tensor]:
    name = "COProblems/maxsat/jnh"+current_list_of_file[idx]+".sat"
    
    # Opening .txt file to read raw data of an instance
    file = open(str(name), 'r', encoding="utf-8")
    x = []
    for line in file:
        split_line = line.split()
        for i in range(len(split_line)):
            x.append(split_line[i])
    file.close()
    
    nb_variables = int(x.pop(0))
    nb_clauses = int(x.pop(0))

    weights = []
    
    clauses = torch.zeros((nb_clauses,nb_variables),dtype=torch.int)
    
    counter = int(x.pop(0))
    weights += [int(x.pop(0))]
    idx_clause = 0
    
    while len(x)>0:
        if counter == 0:
            counter = int(x.pop(0))
            weights += [int(x.pop(0))]
            idx_clause += 1
            
        else :
            val = x.pop(0)
            if int(val) > 0:
                clauses[idx_clause][abs(int(val))-1] = 1
            else:
                clauses[idx_clause][abs(int(val))-1] = -1
            counter -= 1

    Weights = torch.Tensor(weights)
    
    return Weights, clauses
    

partial_solutions = [420925, '4', '5', '6', '7', '8', '9', 420840,
 420753, 420925, 420816, 420824, 420719, 420919, 420925, 420795, 420759,
 394238, 394170, 393881, 393063, 394238, 394159, 394238, 394238, 
 393979, 394238, 394163, 394150, 394226, 394238, 394238, 394156, 394238,
 444854, 444459, 444503, 444533, 444112, 444838, 444314, 444724, 444578,  444391]

current_solutions = [420925, 420840,
 420753, 420925, 420816, 420824, 420719, 420919, 420925, 420795, 420759,
 394238, 394170, 393881, 394063, 394238, 394159, 394238, 394238, 
 393979, 394238, 394163, 394150, 394226, 394238, 394238, 394156, 394238,
 444854, 444459, 444503, 444533, 444112, 444838, 444314, 444724, 444578,  444391]

def SATFitness(idx: int) -> int:

    solution = int(current_solutions[idx])
    return solution