import torch.nn as nn
from thop import profile
from modules import Transformermodule

model = Transformermodule.return_Transformer()
model = model.cuda()
model = nn.DataParallel(model,device_ids=None)
print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters if p.requires_grad)
print('Total number of trainable parameters: ', pytorch_total_params)

flops, prms = profile(model,input_size=(1,3,16,112,112))
print('Total number of FLOPs: ', flops)