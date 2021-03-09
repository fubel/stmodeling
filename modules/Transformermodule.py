import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertModel,BertTokenizer
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

class Transformermodule(nn.Module):
	def __init__(self,image_feature_dim,num_segments,num_class,fc_dim=1024):
		super(Transformermodule,self).__init__()
		self.image_feature_dim = image_feature_dim
		self.num_segments = num_segments
		
		self.configname = 'bert-base-uncased'
		
		# Load Bert Model as the transformer
		self.tokenizer = BertTokenizer.from_pretrained(self.configname)
		self.config = BertConfig.from_pretrained(self.configname)
		self.transformer = BertModel.from_pretrained(self.configname, config = self.config)

		# Project the video embedding to the transformer embedding for processing.
		self.hidden_dim = self.transformer.config.hidden_size

		self.projection_layer = nn.Linear(image_feature_dim,self.hidden_dim)
		self.embedding_fn = BertEmbeddings(self.config)
		self.fc = nn.Sequential(nn.Linear(self.hidden_dim,fc_dim),
								nn.Dropout(0.5),
								nn.Tanh(),
								nn.Linear(fc_dim,num_class))

	def forward(self,input):	
		# Size: [batch size, number of frames, number of features per frame]
		batch_size = input.size()[0]

		# Get attention mask and token sequence ids
		token_seq_ids, attention_mask = self.get_seq_info(batch_size)
		
		projected_frames = self.projection_layer(input)
		
		embedded_frames = self.embedding_fn(inputs_embeds=projected_frames, 
											position_ids=torch.arange(1, self.max_len + 1, device=self.transformer.weight.device)
		

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
	
	def get_seq_info(self,batch_size):
		# [CLS] [UNK] * NUM_FRAMES(NUM_SEGMENTS)
		token_seq_ids = [self.tokenizer.cls_token_id]
		token_seq_ids += [self.tokenizer.unk_token_id] * self.num_segments
		token_seq_ids = np.tile(token_seq_ids,(batch_size,1))

		# Initialize attention mask with 1s. Number of frames + number of special tokens
		attention_mask = np.ones((batch_size,self.num_segments+1))

		print(token_seq_ids)
		# Convert to tensors
		token_seq_ids = torch.tensor(token_seq_ids)
		attention_mask = torch.tensor(attention_mask)
		return token_seq_ids,attention_mask 

def return_Transformer(relation_type, img_feature_dim, num_frames, num_class,fc_dim=1024):
    Transformermodel = Transformermodule(img_feature_dim, num_frames, num_class, fc_dim)

    return Transformermodel
