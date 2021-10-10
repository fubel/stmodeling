import torch.nn as nn
from thop import profile
from modules import Transformermodule

# Jester num_classes: 27 
# SthSthV2 num_classes: 174

model = Transformermodule.return_Transformer('Transformer',512,16,27)
model = model.cuda()
model = nn.DataParallel(model,device_ids=None).cuda()
print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters: {:.2f}M".format(pytorch_total_params/1.e6))

# Input size: [batch size, number of frames, number of features per frame]
flops, prms = profile(model,input_size=(1,16,512))
print('Total number of FLOPs: ', flops)
print("Total number of MFLOPs: {:.2f}".format(flops/1.e6))
