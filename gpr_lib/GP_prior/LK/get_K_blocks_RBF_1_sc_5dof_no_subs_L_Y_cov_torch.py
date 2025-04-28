import torch

def get_K_blocks_RBF_1_sc_5dof_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):

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
    q_i5 = X1[:,pos_indices[4]:pos_indices[4]+1]
    dq_i5 = X1[:,vel_indices[4]:vel_indices[4]+1]
    ddq_i5 = X1[:,acc_indices[4]:acc_indices[4]+1]
    q_j5 = X2[:,pos_indices[4]:pos_indices[4]+1].transpose(1,0)
    dq_j5 = X2[:,vel_indices[4]:vel_indices[4]+1].transpose(1,0)
    ddq_j5 = X2[:,acc_indices[4]:acc_indices[4]+1].transpose(1,0)

    lL1 = lL[0]
    lL6 = lL[5]
    lL11 = lL[10]
    lL2 = lL[1]
    lL7 = lL[6]
    lL12 = lL[11]
    lL3 = lL[2]
    lL8 = lL[7]
    lL13 = lL[12]
    lL4 = lL[3]
    lL9 = lL[8]
    lL14 = lL[13]
    lL5 = lL[4]
    lL10 = lL[9]
    lL15 = lL[14]

    
    x0 = torch.sin(q_j1)
    x1 = lL1**(-2.0)
    x2 = torch.cos(q_j1)
    x3 = -x2 + torch.cos(q_i1)
    x4 = x0*x1*x3
    x5 = lL6**(-2.0)
    x6 = -x0 + torch.sin(q_i1)
    x7 = x2*x5*x6
    x8 = lL12**(-2.0)
    x9 = ddq_j2*x8
    x10 = dq_i2 - dq_j2
    x11 = dq_i1 - dq_j1
    x12 = lL11**(-2.0)
    x13 = 2*x12
    x14 = x11*x13
    x15 = x10*x14
    x16 = dq_i3 - dq_j3
    x17 = lL13**(-2.0)
    x18 = ddq_j3*x17
    x19 = x16*x18
    x20 = dq_i4 - dq_j4
    x21 = lL14**(-2.0)
    x22 = ddq_j4*x21
    x23 = x20*x22
    x24 = dq_i5 - dq_j5
    x25 = lL15**(-2.0)
    x26 = ddq_j5*x25
    x27 = x24*x26
    x28 = dq_j1*(-x4 + x7)
    x29 = torch.cos(q_j2)
    x30 = lL7**(-2.0)
    x31 = torch.sin(q_j2)
    x32 = -x31 + torch.sin(q_i2)
    x33 = x29*x30*x32
    x34 = lL2**(-2.0)
    x35 = -x29 + torch.cos(q_i2)
    x36 = x31*x34*x35
    x37 = dq_j2*(x33 - x36)
    x38 = torch.cos(q_j3)
    x39 = lL8**(-2.0)
    x40 = torch.sin(q_j3)
    x41 = -x40 + torch.sin(q_i3)
    x42 = x38*x39*x41
    x43 = lL3**(-2.0)
    x44 = -x38 + torch.cos(q_i3)
    x45 = x40*x43*x44
    x46 = dq_j3*(x42 - x45)
    x47 = torch.cos(q_j4)
    x48 = lL9**(-2.0)
    x49 = torch.sin(q_j4)
    x50 = -x49 + torch.sin(q_i4)
    x51 = x47*x48*x50
    x52 = lL4**(-2.0)
    x53 = -x47 + torch.cos(q_i4)
    x54 = x49*x52*x53
    x55 = dq_j4*(x51 - x54)
    x56 = torch.sin(q_j5)
    x57 = lL5**(-2.0)
    x58 = torch.cos(q_j5)
    x59 = -x58 + torch.cos(q_i5)
    x60 = lL10**(-2.0)
    x61 = -x56 + torch.sin(q_i5)
    x62 = x56*x57*x59 - x58*x60*x61
    x63 = dq_j5*x62
    x64 = 2*sL*torch.exp(-x1*x3**2 - x10**2*x8 - x11**2*x12 - x16**2*x17 - x20**2*x21 - x24**2*x25 - x30*x32**2 - x34*x35**2 - x39*x41**2 - x43*x44**2 - x48*x50**2 - x5*x6**2 - x52*x53**2 - x57*x59**2 - x60*x61**2)
    x65 = 2*x8
    x66 = x10*x65
    x67 = x16*x17
    x68 = ddq_j1*x14
    x69 = ddq_j2*x66
    x70 = 2*x17
    x71 = x16*x70
    x72 = x20*x21
    x73 = ddq_j3*x71
    x74 = 2*x21
    x75 = x20*x74
    x76 = 2*x25
    x77 = x24*x76
    x78 = x24*x25
    
    K_block_list = []
    K_block_list.append(x64*(-ddq_j1*x12*(-x11**2*x13 + 1) + x14*x19 + x14*x23 + x14*x27 + x14*x28 + x14*x37 + x14*x46 + x14*x55 - x14*x63 + x15*x9 + x4 - x7))
    K_block_list.append(x64*(ddq_j1*x15*x8 + x19*x66 + x23*x66 + x27*x66 + x28*x66 - x33 + x36 + x37*x66 + x46*x66 + x55*x66 - x63*x66 - x9*(-x10**2*x65 + 1)))
    K_block_list.append(x64*(-x18*(-x16**2*x70 + 1) + x23*x71 + x27*x71 + x28*x71 + x37*x71 - x42 + x45 + x46*x71 + x55*x71 - x63*x71 + x67*x68 + x67*x69))
    K_block_list.append(x64*(-x22*(-x20**2*x74 + 1) + x27*x75 + x28*x75 + x37*x75 + x46*x75 - x51 + x54 + x55*x75 - x63*x75 + x68*x72 + x69*x72 + x72*x73))
    K_block_list.append(x64*(ddq_j4*x75*x78 - x26*(-x24**2*x76 + 1) + x28*x77 + x37*x77 + x46*x77 + x55*x77 + x62 - x63*x77 + x68*x78 + x69*x78 + x73*x78))
    
    return K_block_list

