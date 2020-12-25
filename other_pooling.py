import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixed(nn.Module):
    """Custom layer for Mixed pooling 
    Arguments:
        kernel_size   : Single integer denoting the dimension of the square kernel
        stride        : Single integer denoting the equal stride in both directions
    Returns:
        mix           : Mixed pooling between max and avg pooling
    """
    def __init__(self, kernel_size, stride):
        super(Mixed, self).__init__()
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        alpha = torch.Tensor(1)
        self.alpha = nn.Parameter(alpha, requires_grad=True)

        nn.init.uniform_(self.alpha, a=0.25, b=0.75)

    def forward(self, x):
        mix = self.alpha*self.max(x) + (1-self.alpha)*self.avg(x)
        return mix


class Gated(nn.Module):
    """Custom Layer for Gated pooling, with one gate for all windows across all dimensions
    Arguments:
        kernel_size   : Single integer denoting the dimension of the square kernel
        stride        : Single integer denoting the equal stride in both directions
    Returns:
        gated         : Gated pooling between max and avg pooling, with a single gate for each window
    """
    def __init__(self, kernel_size, stride):
        super(Gated, self).__init__()
        self.pool = kernel_size
        self.stride = stride
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        mask = torch.Tensor(1, 1, self.pool, self.pool)
        self.mask = nn.Parameter(mask, requires_grad=True)

        nn.init.normal_(self.mask, mean=0.0, std=1.0)

    def forward(self, x):
        self.batch, self.channels, self.row, self.column = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # Changing the dimensions from (batch, channels, row, column) to (batch*channels, 1, row, column)
        # Allows us to use the standard convolution operation with the weight being same across all the channels, ie, 
        # Using an esentially 2-D weight instead of a three dimensional one

        z = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.mask, bias=None, stride=(stride, stride)).view(self.batch, self.channels, self.row//2, self.column//2)
        z = torch.sigmoid(z)
        gated = torch.add(torch.mul(self.max(x), z), torch.mul(self.avg(x), (1-z)))
        return gated


class Tree_level2(nn.Module):
    """Custom layer for Tree based pooling of tree level-2
    Arguments:
        kernel_size   : Single integer denoting the dimension of the square kernel
        stride        : Single integer denoting the equal stride in both directions
    Returns:
        output        : Tree pooling of depth 2 of input layer
    """
    
    def __init__(self, kernel_size, stride):
        super(Tree_level2, self).__init__()
        self.pool = kernel_size
        self.s = stride
        
        v1 = torch.Tensor(1, 1, self.pool, self.pool)
        v2 = torch.Tensor(1, 1, self.pool, self.pool)
        w3 = torch.Tensor(1, 1, self.pool, self.pool)
        self.v1 = nn.Parameter(v1, requires_grad=True)
        self.v2 = nn.Parameter(v2, requires_grad=True)
        self.w3 = nn.Parameter(w3, requires_grad=True)

        nn.init.uniform_(self.v1, a=0.25, b=0.75)
        nn.init.uniform_(self.v2, a=0.25, b=0.75)
        nn.init.uniform_(self.w3, a=0.25, b=0.75)

    def forward(self, x):
        self.batch, self.channels, self.row, self.column = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        leaf1 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.v1, bias=None, stride=(self.s, self.s)).view(self.batch, self.channels, self.row//2, self.column//2)
        leaf2 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.v2, bias=None, stride=(self.s, self.s)).view(self.batch, self.channels, self.row//2, self.column//2)
        root = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.w3, bias=None, stride=(self.s, self.s)).view(self.batch, self.channels, self.row//2, self.column//2)
        
        torch.sigmoid_(root)
        output = torch.add(torch.mul(leaf1, root), torch.mul(leaf2, (1-root)))
        return output


class Tree_level3(nn.Module):
    """Custom layer for Tree based pooling of tree level-3
    Arguments:
        kernel_size   : Single integer denoting the dimension of the square kernel
        stride        : Single integer denoting the equal stride in both directions
    Returns:
        output        : Tree pooling of depth 2 of input layer
    """
    def __init__(self, kernel_size, stride):
        super(Tree_level3, self).__init__()
        self.pool = kernel_size
        self.s = stride

        v1 = torch.Tensor(1, 1, self.pool, self.pool)
        v2 = torch.Tensor(1, 1, self.pool, self.pool)
        v3 = torch.Tensor(1, 1, self.pool, self.pool)
        v4 = torch.Tensor(1, 1, self.pool, self.pool)
        w5 = torch.Tensor(1, 1, self.pool, self.pool)
        w6 = torch.Tensor(1, 1, self.pool, self.pool)
        w7 = torch.Tensor(1, 1, self.pool, self.pool)
        self.v1 = nn.Parameter(v1, requires_grad=True)
        self.v2 = nn.Parameter(v2, requires_grad=True)
        self.v3 = nn.Parameter(v1, requires_grad=True)
        self.v4 = nn.Parameter(v1, requires_grad=True)
        self.w5 = nn.Parameter(w5, requires_grad=True)
        self.w6 = nn.Parameter(w6, requires_grad=True)
        self.w7 = nn.Parameter(w7, requires_grad=True)

        nn.init.uniform_(self.v1, a=0.25, b=0.75)
        nn.init.uniform_(self.v2, a=0.25, b=0.75)
        nn.init.uniform_(self.v3, a=0.25, b=0.75)
        nn.init.uniform_(self.v4, a=0.25, b=0.75)
        nn.init.uniform_(self.w5, a=0.25, b=0.75)
        nn.init.uniform_(self.w6, a=0.25, b=0.75)
        nn.init.uniform_(self.w7, a=0.25, b=0.75)

    def forward(self, x):
        self.batch, self.channels, self.row, self.column = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        leaf1 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.v1, bias=None, stride=self.s).view(self.batch, self.channels, self.row//2, self.column//2)
        leaf2 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.v2, bias=None, stride=self.s).view(self.batch, self.channels, self.row//2, self.column//2)
        leaf3 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.v3, bias=None, stride=self.s).view(self.batch, self.channels, self.row//2, self.column//2)
        leaf4 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.v4, bias=None, stride=self.s).view(self.batch, self.channels, self.row//2, self.column//2)
        internal5 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.w5, bias=None, stride=self.s).view(self.batch, self.channels, self.row//2, self.column//2)
        internal6 = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.w6, bias=None, stride=self.s).view(self.batch, self.channels, self.row//2, self.column//2)
        root = F.conv2d(input=x.view(self.batch*self.channels, 1, self.row, self.column), weight=self.w7, bias=None, stride=self.s).view(self.batch, self.channels, self.row//2, self.column//2)
        
        torch.sigmoid_(internal5)
        torch.sigmoid_(internal6)
        torch.sigmoid_(root)

        node5 = torch.add(torch.mul(leaf1, internal5), torch.mul(leaf2, (1-internal5)))
        node6 = torch.add(torch.mul(leaf3, internal6), torch.mul(leaf4, (1-internal6)))
        output = torch.add(torch.mul(node5, root), torch.mul(node6, (1-root)))
        return output