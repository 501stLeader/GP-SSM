import torch

def get_K_blocks_RBF_1_sc_4dof_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):

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
    q_i4 = X1[:,pos_indices[3]:pos_indices[3]+1]
    dq_i4 = X1[:,vel_indices[3]:vel_indices[3]+1]
    ddq_i4 = X1[:,acc_indices[3]:acc_indices[3]+1]
    q_j4 = X2[:,pos_indices[3]:pos_indices[3]+1].transpose(1,0)
    dq_j4 = X2[:,vel_indices[3]:vel_indices[3]+1].transpose(1,0)
    ddq_j4 = X2[:,acc_indices[3]:acc_indices[3]+1].transpose(1,0)

    lL1 = lL[0]
    lL5 = lL[4]
    lL9 = lL[8]
    lL2 = lL[1]
    lL6 = lL[5]
    lL10 = lL[9]
    lL3 = lL[2]
    lL7 = lL[6]
    lL11 = lL[10]
    lL4 = lL[3]
    lL8 = lL[7]
    lL12 = lL[11]

    
    x0 = torch.sin(q_j1)
    x1 = lL1**(-2.0)
    x2 = torch.cos(q_j1)
    x3 = -x2 + torch.cos(q_i1)
    x4 = x0*x1*x3
    x5 = lL5**(-2.0)
    x6 = -x0 + torch.sin(q_i1)
    x7 = x2*x5*x6
    x8 = lL10**(-2.0)
    x9 = ddq_j2*x8
    x10 = dq_i2 - dq_j2
    x11 = dq_i1 - dq_j1
    x12 = lL9**(-2.0)
    x13 = 2*x12
    x14 = x11*x13
    x15 = x10*x14
    x16 = dq_i3 - dq_j3
    x17 = lL11**(-2.0)
    x18 = ddq_j3*x17
    x19 = x16*x18
    x20 = dq_i4 - dq_j4
    x21 = lL12**(-2.0)
    x22 = ddq_j4*x21
    x23 = x20*x22
    x24 = dq_j1*(-x4 + x7)
    x25 = torch.cos(q_j2)
    x26 = lL6**(-2.0)
    x27 = torch.sin(q_j2)
    x28 = -x27 + torch.sin(q_i2)
    x29 = x25*x26*x28
    x30 = lL2**(-2.0)
    x31 = -x25 + torch.cos(q_i2)
    x32 = x27*x30*x31
    x33 = dq_j2*(x29 - x32)
    x34 = torch.cos(q_j3)
    x35 = lL7**(-2.0)
    x36 = torch.sin(q_j3)
    x37 = -x36 + torch.sin(q_i3)
    x38 = x34*x35*x37
    x39 = lL3**(-2.0)
    x40 = -x34 + torch.cos(q_i3)
    x41 = x36*x39*x40
    x42 = dq_j3*(x38 - x41)
    x43 = torch.cos(q_j4)
    x44 = lL8**(-2.0)
    x45 = torch.sin(q_j4)
    x46 = -x45 + torch.sin(q_i4)
    x47 = x43*x44*x46
    x48 = lL4**(-2.0)
    x49 = -x43 + torch.cos(q_i4)
    x50 = x45*x48*x49
    x51 = dq_j4*(x47 - x50)
    x52 = 2*sL*torch.exp(-x1*x3**2 - x10**2*x8 - x11**2*x12 - x16**2*x17 - x20**2*x21 - x26*x28**2 - x30*x31**2 - x35*x37**2 - x39*x40**2 - x44*x46**2 - x48*x49**2 - x5*x6**2)
    x53 = 2*x8
    x54 = x10*x53
    x55 = x16*x17
    x56 = ddq_j1*x14
    x57 = ddq_j2*x54
    x58 = 2*x17
    x59 = x16*x58
    x60 = x20*x21
    x61 = 2*x21
    x62 = x20*x61
    
    K_block_list = []
    K_block_list.append(x52*(-ddq_j1*x12*(-x11**2*x13 + 1) + x14*x19 + x14*x23 + x14*x24 + x14*x33 + x14*x42 + x14*x51 + x15*x9 + x4 - x7))
    K_block_list.append(x52*(ddq_j1*x15*x8 + x19*x54 + x23*x54 + x24*x54 - x29 + x32 + x33*x54 + x42*x54 + x51*x54 - x9*(-x10**2*x53 + 1)))
    K_block_list.append(x52*(-x18*(-x16**2*x58 + 1) + x23*x59 + x24*x59 + x33*x59 - x38 + x41 + x42*x59 + x51*x59 + x55*x56 + x55*x57))
    K_block_list.append(x52*(ddq_j3*x59*x60 - x22*(-x20**2*x61 + 1) + x24*x62 + x33*x62 + x42*x62 - x47 + x50 + x51*x62 + x56*x60 + x57*x60))
    
    return K_block_list

