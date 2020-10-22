from helper import *

class AcrE(torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	
	Returns
	-------
	The AcrE model instance
		
	"""
	def __init__(self, params):
		super(AcrE, self).__init__()

		self.p                  = params
		self.ent_embed		= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None); xavier_normal_(self.rel_embed.weight)
		self.bceloss		= torch.nn.BCELoss()

		self.inp_drop		= torch.nn.Dropout(self.p.inp_drop)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.feature_map_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(self.p.channel)
		self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
		self.fc = torch.nn.Linear(self.p.channel * 400, self.p.embed_dim)
		self.padding 		= 0
		self.way = self.p.way
		self.first_atrous	= self.p.first_atrous
		self.second_atrous  = self.p.second_atrous
		self.third_atrous	= self.p.third_atrous

		if self.way == 's':
			self.conv1 = torch.nn.Conv2d(1, self.p.channel, (3, 3), 1, self.first_atrous, bias=self.p.bias, dilation=self.first_atrous)
			self.conv2 = torch.nn.Conv2d(self.p.channel, self.p.channel, (3, 3), 1, self.second_atrous, bias=self.p.bias, dilation=self.second_atrous)
			self.conv3 = torch.nn.Conv2d(self.p.channel, self.p.channel, (3, 3), 1, self.third_atrous, bias=self.p.bias, dilation=self.third_atrous)
		else:
			self.conv1 = torch.nn.Conv2d(1, self.p.channel, (3, 3), 1, self.first_atrous, bias=self.p.bias,
										 dilation=self.first_atrous)
			self.conv2 = torch.nn.Conv2d(1, self.p.channel, (3, 3), 1, self.second_atrous, bias=self.p.bias,
										 dilation=self.second_atrous)
			self.conv3 = torch.nn.Conv2d(1, self.p.channel, (3, 3), 1, self.third_atrous, bias=self.p.bias,
										 dilation=self.third_atrous)
			self.W_gate_e = torch.nn.Linear(1600, 400)

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0];
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def forward(self, sub, rel, neg_ents, strategy='one_to_x'):
		sub_emb		= self.ent_embed(sub).view(-1, 1, 10, 20)
		rel_emb		= self.rel_embed(rel).view(-1, 1, 10, 20)
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=2)
		stack_inp = self.bn0(comb_emb)
		x		= self.inp_drop(stack_inp)
		res = x
		if self.way == 's':
			x = self.conv1(x)
			x = self.conv2(x)
			x = self.conv3(x)
			x = x + res
		else:
			conv1 = self.conv1(x).view(-1, self.p.channel, 400)
			conv2 = self.conv2(x).view(-1, self.p.channel, 400)
			conv3 = self.conv3(x).view(-1, self.p.channel, 400)
			res = res.expand(-1, self.p.channel, 20, 20).view(-1, self.p.channel, 400)
			x = torch.cat((res, conv1, conv2, conv3), dim=2)
			x = self.W_gate_e(x).view(-1, self.p.channel, 20, 20)
		x		= self.bn1(x)
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x		= x.view(x.shape[0], -1)
		x		= self.fc(x)
		x		= self.hidden_drop(x)
		x		= self.bn2(x)
		x		= F.relu(x)

		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]

		pred	= torch.sigmoid(x)

		return pred
