# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

import torch.nn as nn
import torch.nn.functional as F

                                             
                                  
class CoordNet(nn.Module):
                                                      
    def __init__(self, args):
        thought_dim = args.rnn_hidden_dim
        super(CoordNet, self).__init__()

        self.hidden_dim = thought_dim // 2
        self.FC1 = nn.Linear(thought_dim, self.hidden_dim)
        self.BN1 = nn.LayerNorm(self.hidden_dim)
                                                    
        self.bicnet = nn.GRU(input_size=self.hidden_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=1,
                             batch_first=True,
                                                  
                             bidirectional=True)
                                                       
    '''
    thought_comb : n_agent, batch_size, thought_dim
    '''

    def forward(self, thought_comb):
        thought_comb = self.BN1(F.relu(self.FC1(thought_comb)))
        bi_output, _ = self.bicnet(thought_comb)
                                       
        return F.relu(bi_output)

                              
class Q_out(nn.Module):
    def __init__(self, args, hidden_shape=32):
        super(Q_out, self).__init__()
        self.fc1 = nn.Linear(args.rnn_hidden_dim*2, hidden_shape)
        self.fc2 = nn.Linear(hidden_shape, args.n_actions)

    def forward(self, input_mac):
        hidden = F.elu(self.fc1(input_mac))
        output_mac = self.fc2(hidden)
        return output_mac
