import torch
import torch.nn as nn
import torch.nn.functional as F





class ProjectionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Linear(320, 64)
        
        
	def forward(self, x):
		x = self.fc(x)
		
		return x
    
    
    # 与分类器相同架构的投影模型，对表示向量进行下采样，对输出进行归一化，以便再较低维度空间内以更有效的方式计算监督对比损失
