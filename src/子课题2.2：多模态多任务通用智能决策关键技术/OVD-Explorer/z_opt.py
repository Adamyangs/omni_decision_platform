# no longer use.
import torch
import torch.optim as optim
import utils.pytorch_util as ptu
from torch.autograd import Variable

class Z_opt(object):
    def __init__(self):
        super().__init__()
        self.z_auto = torch.tensor([0.5]).cuda()
        # self.z_auto = ptu.zeros(1, requires_grad=True).cuda()
        # self.z_auto = self.z_auto  + 0.5
        # self.z_auto = Variable(torch.randn(1)).cuda()
        self.z_auto.requires_grad = True
        self.z_optimizer = optim.Adam([self.z_auto], lr=3e-4)
    
    def get_z(self):
        return self.z_auto.detach().cpu().numpy().tolist()[0]
    
    def train(self, mu, std, action, g_cdf):
        g_cdf = g_cdf.detach() * self.z_auto.detach() / self.z_auto
        a_dis = torch.distributions.Normal(mu, std)
        log_pi_a_e = a_dis.log_prob(action.detach())
        # z_loss = mu * self.z_auto
        z_loss = -(g_cdf - 1) * log_pi_a_e - g_cdf * torch.log(g_cdf)
        z_loss = z_loss.mean()
        # print(g_cdf, g_pdf, grad, gradmean, gradstd)
        # print("action, log_pi_a_e, z_loss, z_auto, g_cdf", action, log_pi_a_e, z_loss, self.z_auto, g_cdf)
        # print("", self.z_auto.grad)
        print("z, z_loss", self.z_auto.detach().cpu().numpy().tolist()[0], z_loss)
        self.z_optimizer.zero_grad()
        z_loss.backward()
        self.z_optimizer.step()

