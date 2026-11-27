import torch



def cross_entropy(input, target):
    max_term = torch.max(input, -1, keepdim=True).values
    input = input - max_term
    denominator = torch.log(torch.sum(torch.exp(input), -1,keepdim=True))
    target = target.unsqueeze(-1)
    return torch.mean(- torch.gather(input, -1, target)  +  denominator)



