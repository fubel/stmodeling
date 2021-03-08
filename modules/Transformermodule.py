import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from transformers.modeling_bert import BertEmbeddings

class Transformermodule(nn.Module):
	def __init__(self,input_dim,hidden_dim,fc_dim,num_class,num_segments):
		super(Transformermodule,self).__init__()
		input_dim = input_dim

		self.num_segments = num_segments

		# Load Bert Model as the transformer
		self.transformer = BertModel

		# Project the video embedding to the transformer embedding for processing.
		self.projection_layer = nn.Linear(input_dim,hidden_dim)

		self.embedding_fn = BertEmbeddings

		self.fc = nn.Sequential(nn.Linear(hidden_dim,fc_dim),
								nn.Dropout(0.5),
								nn.Tanh(),
								nn.Linear(fc_dim,num_class))

	def forward(self,frames,token_seq_ids,attention_mask):
		
		# Size: [batch size, number of frames, number of features per frame]
		projected_frames = self.projection_layer(video)
		embedded_frames = self.embedding_fn(input_embeds=frames,
									position_ids = torch.arange(1,self.num_segments + 1))

		embeddings_input = self.embedding_fn(input_ids=token_seq_ids)
		embeddings_input[:,1:self.num_segments+1,:] = frames
		tranformer_output = self.transformer(inputs_embeds=embeddings_input,           # [batch, max_len, emb_dim]
                                              attention_mask=attention_mask)[0]

		# Take the first value
		classes_output = transformer_output[:,0,:]
		logits = self.fc(classes_output)

		return logits