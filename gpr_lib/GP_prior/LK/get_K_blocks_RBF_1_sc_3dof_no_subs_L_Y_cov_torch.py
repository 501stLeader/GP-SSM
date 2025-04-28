import torch

def get_K_blocks_RBF_1_sc_3dof_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):

    q_i1 = X1[:,pos_indices[0]:pos_indices[0]+1]
    dq_i1 = X1[:,vel_indices[0]:vel_indices[0]+1]
    ddq_i1 = X1[:,acc_indices[0]:acc_indices[0]+1]
    q_j1 = X2[:,pos_indices[0]:pos_indices[0]+1].transpose(1,0)
    dq_j1 = X2[:,vel_indices[0]:vel_indices[0]+1].transpose(1,0)
    ddq_j1 = X2[:,acc_indices[0]:acc_indices[0]+1].transpose(1,0)
    q_i2 = X1[:,pos_indices[1]:pos_indices[1]+1]
    dq_i2 = X1[:,vel_indices[1]:vel_indices[1]+1]
    ddq_i2 = X1[:,acc_indices[1]:acc_indices[1]+1]
    q_j2 = X2[:,pos_indices[1]:pos_indices[1]+1].transpose(1,0)
    dq_j2 = X2[:,vel_indices[1]:vel_indices[1]+1].transpose(1,0)
    ddq_j2 = X2[:,acc_indices[1]:acc_indices[1]+1].transpose(1,0)
    q_i3 = X1[:,pos_indices[2]:pos_indices[2]+1]
    dq_i3 = X1[:,vel_indices[2]:vel_indices[2]+1]
    ddq_i3 = X1[:,acc_indices[2]:acc_indices[2]+1]
    q_j3 = X2[:,pos_indices[2]:pos_indices[2]+1].transpose(1,0)
    dq_j3 = X2[:,vel_indices[2]:vel_indices[2]+1].transpose(1,0)
    ddq_j3 = X2[:,acc_indices[2]:acc_indices[2]+1].transpose(1,0)

    lL1 = lL[0]
    lL4 = lL[3]
    lL7 = lL[6]
    lL2 = lL[1]
    lL5 = lL[4]
    lL8 = lL[7]
    lL3 = lL[2]
    lL6 = lL[5]
    lL9 = lL[8]

    
    x0 = torch.sin(q_j1)
    x1 = lL1**(-2.0)
    x2 = torch.cos(q_j1)
    x3 = -x2 + torch.cos(q_i1)
    x4 = x0*x1*x3
    x5 = lL4**(-2.0)
    x6 = -x0 + torch.sin(q_i1)
    x7 = x2*x5*x6
    x8 = lL8**(-2.0)
    x9 = ddq_j2*x8
    x10 = dq_i2 - dq_j2
    x11 = dq_i1 - dq_j1
    x12 = lL7**(-2.0)
    x13 = 2*x12
    x14 = x11*x13
    x15 = x10*x14
    x16 = dq_i3 - dq_j3
    x17 = lL9**(-2.0)
    x18 = ddq_j3*x17
    x19 = x16*x18
    x20 = dq_j1*(-x4 + x7)
    x21 = torch.cos(q_j2)
    x22 = lL5**(-2.0)
    x23 = torch.sin(q_j2)
    x24 = -x23 + torch.sin(q_i2)
    x25 = x21*x22*x24
    x26 = lL2**(-2.0)
    x27 = -x21 + torch.cos(q_i2)
    x28 = x23*x26*x27
    x29 = dq_j2*(x25 - x28)
    x30 = torch.cos(q_j3)
    x31 = lL6**(-2.0)
    x32 = torch.sin(q_j3)
    x33 = -x32 + torch.sin(q_i3)
    x34 = x30*x31*x33
    x35 = lL3**(-2.0)
    x36 = -x30 + torch.cos(q_i3)
    x37 = x32*x35*x36
    x38 = dq_j3*(x34 - x37)
    x39 = 2*sL*torch.exp(-x1*x3**2 - x10**2*x8 - x11**2*x12 - x16**2*x17 - x22*x24**2 - x26*x27**2 - x31*x33**2 - x35*x36**2 - x5*x6**2)
    x40 = 2*x8
    x41 = x10*x40
    x42 = x16*x17
    x43 = 2*x17
    x44 = x16*x43
    
    K_block_list = []
    K_block_list.append(x39*(-ddq_j1*x12*(-x11**2*x13 + 1) + x14*x19 + x14*x20 + x14*x29 + x14*x38 + x15*x9 + x4 - x7))
    K_block_list.append(x39*(ddq_j1*x15*x8 + x19*x41 + x20*x41 - x25 + x28 + x29*x41 + x38*x41 - x9*(-x10**2*x40 + 1)))
    K_block_list.append(x39*(ddq_j1*x14*x42 + ddq_j2*x41*x42 - x18*(-x16**2*x43 + 1) + x20*x44 + x29*x44 - x34 + x37 + x38*x44))
    
    return K_block_list

