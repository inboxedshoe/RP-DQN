import torch 


def tim_genius(orig,actions):
    #save the number of column of original
    n = orig.size(1)
    #repeat the rows to create scatterable version
    orig = orig.repeat_interleave(n, dim= 0).view(orig.size(0),-1,orig.size(1))

    #create identity matrix without last row
    mask = torch.eye(n ,n+1).bool().unsqueeze(0)
    mask = mask.repeat_interleave(orig.size(0), dim = 0)
    
    #create placeholder tensor
    a = torch.zeros(orig.size(0),orig.size(1),orig.size(2)+1).float() 
    #create scatterable action matrix
    actions = actions.repeat_interleave(n, dim= 0)
    
    #scatter actions in diagonal, and the values of original everywhere else
    a.masked_scatter_(mask, actions)
    a.masked_scatter_(mask == False, orig)
    
    #copy first column and append it to the end so we have a loop
    a = torch.cat((a, a[:,:,0].view(a.size(0), a.size(1), 1)), dim = 2)
    #unfold into 2 element chunks
    a = a.unfold(2, 2, 1)

    return(a)



trial = torch.tensor([[2,3,2],[3,3,3],[4,4,4],[5,5,5]], dtype = torch.float32)
b = torch.tensor ([10,15,20,25], dtype = torch.float32)

z = tim_genius(trial)

