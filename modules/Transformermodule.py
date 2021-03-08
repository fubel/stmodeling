import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel,BertTokenizer
from transformers.modeling_bert import BertEmbeddings

class Transformermodule(nn.Module):
	def __init__(self,input_dim,hidden_dim,fc_dim,num_class,num_segments):
		super(Transformermodule,self).__init__()
		self.input_dim = input_dim
		self.num_segments = num_segments

		# Load Bert Model as the transformer
		self.transformer = BertModel
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		
		# Project the video embedding to the transformer embedding for processing.
		self.projection_layer = nn.Linear(input_dim,hidden_dim)
		self.embedding_fn = BertEmbeddings
		self.fc = nn.Sequential(nn.Linear(hidden_dim,fc_dim),
								nn.Dropout(0.5),
								nn.Tanh(),
								nn.Linear(fc_dim,num_class))

	def forward(self,video):
		
		# Size: [batch size, number of frames, number of features per frame]
		projected_frames = self.projection_layer(video)
		embedded_frames = self.embedding_fn(input_embeds=projected_frames,
									position_ids = torch.arange(1,self.num_segments + 1))

		# Get attention mask and token sequence ids
		token_seq_ids, attention_mask = get_seq_info()

		embeddings_input = self.embedding_fn(input_ids=token_seq_ids)
		
		# Replace [UNK] embeddings with video embeddings
		embeddings_input[:,1:self.num_segments+1,:] = embedded_frames

		# [batch, max_len, emb_dim]
		tranformer_output = self.transformer(inputs_embeds=embeddings_input,           
                                              attention_mask=attention_mask)[0]

		# Take the first value
		classes_output = transformer_output[:,0,:]
		logits = self.fc(classes_output)

		return logits
	def get_seq_info(self):
		# [CLS] [UNK] * NUM_FRAMES(NUM_SEGMENTS)
		token_seq_ids = [self.tokenizer.cls_token_id]
		token_seq_ids += [self.tokenizer.unk_token_id] * self.num_segments

		# Initialize attention mask with 1s. Number of frames + number of special tokens
		attention_mask = [1] * (self.num_segments + 1)

		# Convert to tensors
		token_seq_ids = torch.tensor(token_seq_ids)
		attention_mask = torch.sensor(attention_mask)
		return token_seq_ids,attention_mask 
