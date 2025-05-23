import numpy as np 
import torch 
import torch.nn.functional as F 
from torch import nn
# from TorchSUL import Model as M 
from torch.nn.parameter import Parameter
import torch.nn.init as init 

#PropLayer 是一个关键点传播和特征变换的基本单元
class PropLayer(nn.Module):
	def __init__(self, num_pts, indim, outdim, usebias=True):
		super(PropLayer, self).__init__()
	
		self.outdim = outdim
		self.act = torch.nn.ReLU()
		self.act2 = torch.nn.ReLU()
		self.usebias = usebias

		self.weight = Parameter(torch.Tensor(num_pts, indim, self.outdim))
		self.weight2 = Parameter(torch.Tensor(num_pts, self.outdim, self.outdim))
		init.kaiming_uniform_(self.weight, a=np.sqrt(5))
		init.kaiming_uniform_(self.weight2, a=np.sqrt(5))

		if self.usebias:
			# print('initialize bias')
			self.bias = Parameter(torch.Tensor(num_pts, self.outdim)) 
			self.bias2 = Parameter(torch.Tensor(num_pts, self.outdim)) 
			init.uniform_(self.bias, -0.1, 0.1)
			init.uniform_(self.bias2, -0.1, 0.1)

	def forward(self, inp, aff=None, act=True):
		if aff is not None:
			# propagate the keypoints 
			x = torch.einsum('ikl,ijk->ijl', inp, aff)
		else:
			x = inp 

		x = torch.einsum('ijk,jkl->ijl', x, self.weight)
		if self.usebias:
			x = x + self.bias
		if act:
			x = self.act(x)
		# x = F.dropout(x, 0.25, self.training, False)

		x = torch.einsum('ijk,jkl->ijl', x, self.weight2)
		if self.usebias:
			x = x + self.bias2
		if act:
			x = self.act2(x)
		#x = F.dropout(x, 0.25, self.training, False)

		if aff is not None:
			x = torch.cat([inp, x], dim=-1)
		return x 

class TransNet(nn.Module):
	def __init__(self, indim, outdim, num_pts):
		super(TransNet, self).__init__()
	
		self.num_pts = num_pts
		self.c1 = PropLayer(num_pts, indim, outdim)
		self.c2 = PropLayer(num_pts, outdim, outdim)
		self.c3 = PropLayer(num_pts, outdim*2, outdim)

		self.b2 = PropLayer(num_pts-1, outdim, outdim)
		self.b3 = PropLayer(num_pts-1, outdim*2, outdim)

		self.c8 = PropLayer(num_pts, 1536, outdim)
		self.c9 = PropLayer(num_pts, outdim, 3)

	def forward(self, x, aff, aff_bone, inc, inc_inv):
		x = feat = self.c1(x)
		x = self.c2(x, aff)
		x = self.c3(x, aff)

		feat = torch.einsum('ijk,lj->ilk', feat, inc)
		feat = self.b2(feat, aff_bone)
		feat = self.b3(feat, aff_bone)
		feat = torch.einsum('ijk,lj->ilk', feat, inc_inv)
		x = torch.cat([x, feat], dim=-1)
		
		x = self.c8(x)
		x = self.c9(x, act=False)
		# print(x.shape)
		x = x.reshape(-1, self.num_pts, 3)
		return x 
