'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch 

entrypoints = torch.hub.list('pytorch/vision:v0.10.0', force_reload=False)
print(entrypoints)