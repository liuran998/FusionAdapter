import logging
import torch.nn as nn
import torch
from params import *
from collections import OrderedDict

class Metanet_text(nn.Module):
    def __init__(self,parameter):
        super(Metanet_text, self).__init__()
        self.params = get_params()
        self.device = parameter['device']
        # self.neuron = parameter['neuron']



        # self.linear = nn.Linear(self.params['embed_dim'], 50)
        # self.bn = nn.BatchNorm2d(num_features = None, affine = False,track_running_stats = False )

        self.MLP1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(1000,50)),
            # ('bn',   nn.BatchNorm2d(num_features = None, affine = False,track_running_stats = False )),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear( 50, 100)),
        ]))

        # self.output = nn.Sequential(OrderedDict([
        #     ('fc',   nn.Linear(self.neuron, self.params['embed_dim'])),
        # ]))

        nn.init.xavier_normal_(self.MLP1.fc.weight)
        nn.init.xavier_normal_(self.MLP1.fc1.weight)

    def forward(self, rel_agg):

        # size = rel_agg.shape
        oupt = self.MLP1(rel_agg).cuda().to(self.device)
        # oupt = self.output(MLP1).to(self.device)

        return oupt

class WayGAN2(object):
    def __init__(self,parameter):
        # self.args = args
        logging.info("Building Metanet...")

        metanet_text = Metanet_text(parameter)
        self.metanet_text = metanet_text

    def getVariables2(self):
        return (self.metanet_text)

    def getWayGanInstance(self):
        return self.waygan2
