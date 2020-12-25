import torch
import torch.nn as nn

class Fuzzy_type2(nn.Module):
    """Custom layer for Type-2 Fuzzy logic based pooling"""
    def __init__(self, kernel_size, stride):
        super(Fuzzy_type2, self).__init__()
        self.pool = kernel_size
        self.n = kernel_size*kernel_size
        self.stride = stride
        self.h = (self.pool*self.pool+1)//2

    def membership(self, x):
        h = self.h
        kmm = x.mean(dim=-1, keepdim=True)

        if (self.pool%2 == 0):
          for k in range(h-1):
            kmm = torch.cat((x[:, :, :, (h-k-1):(h+k+1)].mean(dim=-1, keepdim=True), kmm), dim=-1)
        
        else:
          for k in range(h-1):
            kmm = torch.cat((x[:, :, :, (h-k-1):(h+k)].mean(dim=-1, keepdim=True), kmm), dim=-1)
        
        v_avg = kmm.mean(dim=-1, keepdim=True)
        return kmm, v_avg

    #add the epsilon replacement later
    def var_vec(self, x, v_avg):
        h = self.h
        omega = abs(x - v_avg)
        sigma = omega.mean(dim=-1, keepdim=True)
        
        if(h%2==1):
          for k in range(h-1):
            sigma = torch.cat((omega[:, :, :, (h-k-1):h+k].mean(dim=-1, keepdim=True), sigma), dim=-1)
        else:
          for k in range(h-1):
            sigma = torch.cat((omega[:, :, :, (h-k-1):(h+k+1)].mean(dim=-1, keepdim=True), sigma), dim=-1)
        
        epsilon = 0.0001
        return sigma+epsilon

      
    def delta(self, x, kmm, sigma):
        batch, channels, output_size = self.batch, self.channels, self.output_size
        h = self.h
        n = self.n

        # All the next 4 Tensors are of shape (batch*channels, outoutput_size, output_size, h, n)
        xrep = x.repeat(1, 1, 1, h).view(batch*channels, output_size, output_size, h, n)
        kmmrep = kmm.repeat(1, 1, 1, n).view(batch*channels, output_size, output_size, n, h).transpose(3, 4)
        sigmarep = sigma.repeat(1, 1, 1, n).view(batch*channels, output_size, output_size, n, h).transpose(3, 4)
        pi = torch.exp(-0.5*( ((xrep-kmmrep)/sigmarep)*((xrep-kmmrep)/sigmarep)))

        #Take a special look at the dimensions and keepdim parameter
        max, _ = torch.max(pi, dim=3, keepdim=False)
        thresh, _ = torch.min(max, dim=3, keepdim=True)

        avg_pi = pi.mean(dim=3, keepdim=False)
        return avg_pi, thresh

    def forward(self, x):
        self.batch, self.channels, self.row, self.column = x.shape
        p = self.pool
        s = self.stride
        self.output_size = self.row//p
        #N = round(0.3*p*p)

        #Since the method does not differentiate between two channels of a single image any more that it
        #differentiates between two channels of two separate images, it is beneficial to fold the first
        #two dimensions, ie, batch_size and channels together.
        x = x.view(self.batch*self.channels, self.row, self.column)
        #Extracting the kernels to be operated upon
        #Making x to be of shape (batch*channel, output_row_size, output_column_size, kernel_size*kernel_size)
        x = x.unfold(dimension=1, size=p, step=s).unfold(dimension=2, size=p, step=s)
        x = x.contiguous().view(x.size()[:3] + (-1,))

        #Getting the k-middle means and v_avg.
        #The below functions are tested, they work
        kmm, v_avg = self.membership(x)
        sigma = self.var_vec(x, v_avg)
        avg_pi, thresh = self.delta(x, kmm, sigma)

        pooled = torch.Tensor(self.batch*self.channels, self.output_size, self.output_size, 1)
        
        #Checking the different conditions
        mask_primary = avg_pi[:, :, :, self.h-1].view(self.batch*self.channels, self.output_size, self.output_size, 1)>=thresh
        # The bit-wise & operator creates a tuple of values at the last dimension, one for each of the conditions
        # instead of doing the bit-wise and sum as it is supposed to
        s_condition = (sigma[:, :, :, self.h-1]<0.001).view(self.batch*self.channels, self.output_size, self.output_size, 1)
        mask_secondary = torch.sum(~mask_primary & s_condition, dim=-1, keepdim=True)
        mask_noisy = ~(torch.logical_or(mask_primary, mask_secondary))

        pooled[mask_primary] = x.mean(dim=-1, keepdim=True)[mask_primary]
        pooled[mask_secondary] = v_avg[mask_secondary]

        count = mask_noisy.sum().item()
        region = x[mask_noisy.repeat((1, 1, 1, self.n))].view(count, self.n)
        g = avg_pi[mask_noisy.repeat((1, 1, 1, self.n))].view(count, self.n)
        denoised = torch.mul(g, region).sum(dim=-1, keepdim=True)/g.sum(dim=-1, keepdim=True)
        denoised = denoised.view(count)
        pooled[mask_noisy] = denoised

        pooled = pooled.view(self.batch, self.channels, self.output_size, self.output_size)
        pooled = pooled.contiguous()

        return pooled

def delta(vec, kmm, sigma, h):
  """
  Arguments:
      x     : Neighbourhood tensor of shape (batch*channels, output_size, output_size, kernel_size*kernel_size)
      kmm   : Tensor of k-middle means of shape (batch*channels, output_size, output_size, h)
      sigma : Tensor of h-variances of shape (batch*channels, output_size, output_size, h)
  
  Returns:
      avg_pi : The delta tensor of average pixel intensities corresponding to different Gaussians of shape x
      thresh : Threshold calculated from the pi tensor of shape (batch*channels, output_size, output_size, 1)
  """
  batch, output_size, output_size, n = x.shape

  # All the next 4 Tensors are of shape (batch, outoutput_size, output_size, h, n)
  xrep = x.repeat(1, 1, 1, h).reshape(batch, output_size, output_size, h, n)
  kmmrep = kmm.repeat(1, 1, 1, n).reshape(batch, output_size, output_size, n, h).transpose(3, 4)
  sigmarep = sigma.repeat(1, 1, 1, n).reshape(batch, output_size, output_size, n, h).transpose(3, 4)
  pi = np.exp(-0.5*( ((xrep-kmmrep)/sigmarep)*((xrep-kmmrep)/sigmarep)))

  #Take a special look at the dimensions and keepdim parameter
  max, _ = torch.max(pi, dim=3, keepdim=False)
  thresh, _ = torch.min(max, dim=3, keepdim=True)
  
  avg_pi = pi.mean(dim=3, keepdim=False)
  return avg_pi, thresh

def membership(x, h, p):
    kmm = x.mean(dim=-1, keepdim=True)

    if (p%2 == 0):
      for k in range(h-1):
        kmm = torch.cat((x[:, :, :, (h-k-1):(h+k+1)].mean(dim=-1, keepdim=True), kmm), dim = -1)
    
    else:
      for k in range(h-1):
        kmm = torch.cat((x[:, :, :, (h-k-1):(h+k)].mean(dim=-1, keepdim=True), kmm), dim=-1)

    v_avg = kmm.mean(dim=-1, keepdim=True)
    return kmm, v_avg

def var_vec(x, v_avg, h, p):
  """
  Arguments:
      x       : Neighbourhood tensor of shape (batch*channels, output_size, output_size, kernel_size*kernel_size)
      v_avg   : Average of k-middle means of shape (batch*channels, output_size, output_size, 1)
      h       : Number of variances
      epsilon : Replacement of sigma

  Returns:
      sigma   : Tensor of different variances of shape (batch*channels, output_size, output_size, h)
  """
  omega = abs(x - v_avg)
  sigma = omega.mean(dim=-1, keepdim=True)
  
  if(p%2==1):
    for k in range(h-1):
      sigma = torch.cat((omega[:, :, :, (h-k-1):h+k].mean(dim=-1, keepdim=True), sigma), dim=-1)
  else:
    for k in range(h-1):
      sigma = torch.cat((omega[:, :, :, (h-k-1):(h+k+1)].mean(dim=-1, keepdim=True), sigma), dim=-1)
  
  return sigma + 0.0001