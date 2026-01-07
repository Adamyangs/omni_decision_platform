import torch as th
import torch.nn.functional as F


# enemy = th.arange(32).reshape(2, 4, 4).float()
# agent = th.arange(16).reshape(2, 2, 4).float()
#
# attention_matrix = th.bmm(agent, enemy.transpose(1, 2))
# attention = F.softmax(attention_matrix, dim=-2)
# attention_index = attention.argmax(-1)
#
# print('-'*10 + 'feature' + '-'*10)
# print('enemy: {}'.format(enemy))
# print('agent: {}'.format(agent))
#
# print('-'*10 + 'attention' + '-'*10)
# print('attention: {}'.format(attention))
# print('attention index: {}'.format(attention_index))
#
# attention_index_repeat = attention_index.unsqueeze(-1).repeat(1, 1, 4)
# print(attention_index_repeat)
#
#
# enemy_select = th.gather(enemy, dim=1, index=attention_index_repeat)
# print(enemy_select.shape)



extend_feats = th.arange(96).reshape(2, 2, 6, 4).float()
last_acton = th.tensor([[0, 5], [1, 4]])
last_action_index = last_acton.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 4)
print(extend_feats.shape)
print(last_action_index.shape)

enemy_select = th.gather(extend_feats, dim=-2, index=last_action_index)
print(extend_feats)
print(enemy_select)
print(enemy_select.shape)


