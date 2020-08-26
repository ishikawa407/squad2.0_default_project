from layers import BiDAFAttention
import torch


torch.manual_seed(1)
torch.random.manual_seed(1)

c = torch.randn((2, 5, 10))
q = torch.randn((2, 3, 10))
c_mask = torch.ones((2, 5))
q_mask = torch.ones((2, 5))

attention = BiDAFAttention(hidden_size=10, drop_prob=0)

result = attention(c, q, c_mask, q_mask)
print(result.size())


