import torch

# 65 different files
list_of_file = [
    '41', '42', '43', '44', '45', '46', '47', '48', '49', '410', 
    '51', '52', '53', '54', '55', '56', '57', '58', '59', '510', 
    '61', '62', '63', '64', '65', 'a1', 'a2', 'a3', 'a4', 'a5', 
    'b1', 'b2', 'b3', 'b4', 'b5', 'c1', 'c2', 'c3', 'c4', 'c5', 
    'd1', 'd2', 'd3', 'd4', 'd5', 'e1', 'e2', 'e3', 'e4', 'e5', 
    'f1', 'f2', 'f3', 'f4', 'f5', 'g1', 'g2', 'g3', 'g4', 'g5', 
    'h1', 'h2', 'h3', 'h4', 'h5'
]


def SCPpopulate(idx : int) -> tuple[torch.Tensor, torch.Tensor]:
    name = "COProblems/scp/"+list_of_file[idx]+".txt"
    
    # Opening .txt file to read raw data of an instance
    file = open(str(name), 'r', encoding="utf-8")
    x = []
    for line in file:
        split_line = line.split()
        for i in range(len(split_line)):
            x.append(split_line[i])
    file.close()
    
    size_row = int(x.pop(0))
    size_column = int(x.pop(0))

    costs = torch.Tensor([int(x.pop(0)) for _ in range(size_column)])
    
    A = torch.zeros((size_row,size_column),dtype=torch.int)
    
    counter = int(x.pop(0))
    row = 0
    
    while len(x)>0:
        if counter == 0:
            counter = int(x.pop(0))
            row += 1
            
        else :
            A[row][int(x.pop(0))-1]=1
            counter -= 1
    
    return costs,A
    
def SCPFitness(idx: int) -> int:
    file = open(r"COProblems/scp/solutions.txt", 'r', encoding="utf-8")
    x = []
    for line in file:
        split_line = line.split()
        for i in range(len(split_line)):
            x.append(split_line[i])
    file.close()
    
    solution = int(x[2*idx+1])

    return solution