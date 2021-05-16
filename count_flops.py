import torch.nn as nn
from thop import profile
from modules import Transformermodule

# Jester num_classes: 27 
# SthSthV2 num_classes: 174

model = Transformermodule.return_Transformer('Transformer',512,8,27)
model = model.cuda()
model = nn.DataParallel(model,device_ids=None).cuda()
print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of trainable parameters: ', pytorch_total_params)

# Input size: [batch size, number of frames, number of features per frame]
flops, prms = profile(model,input_size=(1,8,512))
print('Total number of FLOPs: ', flops)