
from copy import deepcopy
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch as th
import torch.nn.functional as F




class EnvUtils(object):
    def __init__(self):
        self.tsne = manifold.TSNE(n_components=2, perplexity=5)

    def transform_tsne(self, self_feats, ally_feats, enemy_feats, plot=False):

        self_feats = self_feats.clone().cpu().detach().numpy().squeeze(-1)
        ally_feats = ally_feats.clone().cpu().detach().numpy()[0]#.squeeze(0)
        enemy_feats = enemy_feats.clone().cpu().detach().numpy()[0]#.squeeze(0)

        tsne_data = self.tsne.fit_transform(np.concatenate((np.expand_dims(self_feats[0], 0), ally_feats, enemy_feats), axis=0))
        tsne_data_min, tsne_data_max = tsne_data.min(0), tsne_data.max(0)
        tsne_data_norm = (tsne_data - tsne_data_min) / (tsne_data_max - tsne_data_min)

        if plot:

            plt.scatter(tsne_data_norm[0, 0], tsne_data_norm[0, 1], s=150, c='r', label='self')
            plt.scatter(tsne_data_norm[1:10, 0], tsne_data_norm[1:10, 1], s=150, c='g', label='ally')
            plt.scatter(tsne_data_norm[10:, 0], tsne_data_norm[10:, 1], s=150,  c='b', label='enemy')

            for i in range(1, 11):
                plt.annotate(str(i), xy=(tsne_data_norm[i-1, 0], tsne_data_norm[i-1, 1]), xytext=(-10, 10), textcoords='offset points')
            for i in range(1, 12):
                plt.annotate(str(i), xy=(tsne_data_norm[i+9, 0], tsne_data_norm[i+9, 1]), xytext=(-10, 10), textcoords='offset points')


            plt.legend(fontsize=16)
            plt.savefig('tsne.pdf')
            plt.show()

    def action_attention_weight(self, state, own_feats, enemy_feats, agent_outputs, plot=False):
        own_feats = own_feats.clone().cpu().detach().unsqueeze(-2)
        enemy_feats = enemy_feats.clone().cpu().detach()

        ally_enemy_relavant = F.softmax(th.bmm(own_feats, enemy_feats.transpose(1,2)), dim=-1).squeeze(-2)
        ally_enemy_relavant = ally_enemy_relavant.numpy()

        state = deepcopy(state)[0]
        ally_pos = []
        enemy_pos = []

        for i in range(10):
            ally_pos.append([state[i * 4 + 2], state[i * 4 + 3]])
        
        for i in range(11):
            enemy_pos.append([state[40+i * 3 + 1], state[40+i * 3 + 2]])

        ally_pos = np.array(ally_pos)
        enemy_pos = np.array(enemy_pos)
        ally_label = [str(i) for i in range(1, 11)]
        enemy_label = [str(i) for i in range(1, 12)]

        relevant_pos = np.zeros((10, 11))
        for i in range(10):
            for j in range(11):
                relevant_pos[i, j] = np.linalg.norm(ally_pos[i]-enemy_pos[j])

        if plot:
            plt.figure(figsize=(12, 5))

            plt.subplot(121)
            sns.heatmap(ally_enemy_relavant)
            plt.xlabel('enemy', fontsize=16 )
            plt.ylabel('ally', fontsize=16)
            plt.xticks(np.arange(0, 11, 1), np.arange(1, 12, 1))
            plt.yticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
            plt.title('attack attention', fontsize=16)


            # plt.subplot(142)
            # plt.scatter(ally_pos[:,0], ally_pos[:,1])
            # plt.scatter(enemy_pos[:,0], enemy_pos[:,1])

            # for i in range(1, 11):
            #     plt.annotate(str(i-1), xy=(ally_pos[i-1, 0], ally_pos[i-1, 1]), xytext=(-10, 10), textcoords='offset points')
            # for i in range(1, 12):
            #     plt.annotate(str(i-1), xy=(enemy_pos[i-1, 0], enemy_pos[i-1, 1]), xytext=(-10, 10), textcoords='offset points')

            plt.subplot(122)
            sns.heatmap(relevant_pos)
            plt.xlabel('enemy', fontsize=16 )
            plt.ylabel('ally', fontsize=16 )
            plt.xticks(np.arange(0, 11, 1), np.arange(1, 12, 1))
            plt.yticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
            plt.title('relevant position', fontsize=16)

            # plt.subplot(144)
            # sns.heatmap(agent_outputs.clone().cpu().detach().numpy().squeeze(0)[:,6:])

            plt.savefig('attack_attention.pdf')
            plt.show()


