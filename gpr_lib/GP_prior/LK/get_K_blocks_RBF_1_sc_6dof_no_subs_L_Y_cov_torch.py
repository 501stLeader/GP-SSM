import torch

def get_K_blocks_RBF_1_sc_6dof_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):

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
    q_i6 = X1[:,pos_indices[5]:pos_indices[5]+1]
    dq_i6 = X1[:,vel_indices[5]:vel_indices[5]+1]
    ddq_i6 = X1[:,acc_indices[5]:acc_indices[5]+1]
    q_j6 = X2[:,pos_indices[5]:pos_indices[5]+1].transpose(1,0)
    dq_j6 = X2[:,vel_indices[5]:vel_indices[5]+1].transpose(1,0)
    ddq_j6 = X2[:,acc_indices[5]:acc_indices[5]+1].transpose(1,0)

    lL1 = lL[0]
    lL7 = lL[6]
    lL13 = lL[12]
    lL2 = lL[1]
    lL8 = lL[7]
    lL14 = lL[13]
    lL3 = lL[2]
    lL9 = lL[8]
    lL15 = lL[14]
    lL4 = lL[3]
    lL10 = lL[9]
    lL16 = lL[15]
    lL5 = lL[4]
    lL11 = lL[10]
    lL17 = lL[16]
    lL6 = lL[5]
    lL12 = lL[11]
    lL18 = lL[17]

    
    x0 = torch.sin(q_j1)
    x1 = lL1**(-2.0)
    x2 = torch.cos(q_j1)
    x3 = -x2 + torch.cos(q_i1)
    x4 = x0*x1*x3
    x5 = lL7**(-2.0)
    x6 = -x0 + torch.sin(q_i1)
    x7 = x2*x5*x6
    x8 = lL14**(-2.0)
    x9 = ddq_j2*x8
    x10 = dq_i2 - dq_j2
    x11 = dq_i1 - dq_j1
    x12 = lL13**(-2.0)
    x13 = 2*x12
    x14 = x11*x13
    x15 = x10*x14
    x16 = dq_i3 - dq_j3
    x17 = lL15**(-2.0)
    x18 = ddq_j3*x17
    x19 = x16*x18
    x20 = dq_i4 - dq_j4
    x21 = lL16**(-2.0)
    x22 = ddq_j4*x21
    x23 = x20*x22
    x24 = dq_i5 - dq_j5
    x25 = lL17**(-2.0)
    x26 = ddq_j5*x25
    x27 = x24*x26
    x28 = dq_i6 - dq_j6
    x29 = lL18**(-2.0)
    x30 = ddq_j6*x29
    x31 = x28*x30
    x32 = dq_j1*(-x4 + x7)
    x33 = torch.cos(q_j2)
    x34 = lL8**(-2.0)
    x35 = torch.sin(q_j2)
    x36 = -x35 + torch.sin(q_i2)
    x37 = x33*x34*x36
    x38 = lL2**(-2.0)
    x39 = -x33 + torch.cos(q_i2)
    x40 = x35*x38*x39
    x41 = dq_j2*(x37 - x40)
    x42 = torch.cos(q_j3)
    x43 = lL9**(-2.0)
    x44 = torch.sin(q_j3)
    x45 = -x44 + torch.sin(q_i3)
    x46 = x42*x43*x45
    x47 = lL3**(-2.0)
    x48 = -x42 + torch.cos(q_i3)
    x49 = x44*x47*x48
    x50 = dq_j3*(x46 - x49)
    x51 = torch.sin(q_j4)
    x52 = lL4**(-2.0)
    x53 = torch.cos(q_j4)
    x54 = -x53 + torch.cos(q_i4)
    x55 = lL10**(-2.0)
    x56 = -x51 + torch.sin(q_i4)
    x57 = x51*x52*x54 - x53*x55*x56
    x58 = dq_j4*x57
    x59 = torch.sin(q_j5)
    x60 = lL5**(-2.0)
    x61 = torch.cos(q_j5)
    x62 = -x61 + torch.cos(q_i5)
    x63 = lL11**(-2.0)
    x64 = -x59 + torch.sin(q_i5)
    x65 = x59*x60*x62 - x61*x63*x64
    x66 = dq_j5*x65
    x67 = torch.sin(q_j6)
    x68 = lL6**(-2.0)
    x69 = torch.cos(q_j6)
    x70 = -x69 + torch.cos(q_i6)
    x71 = lL12**(-2.0)
    x72 = -x67 + torch.sin(q_i6)
    x73 = x67*x68*x70 - x69*x71*x72
    x74 = dq_j6*x73
    x75 = 2*sL*torch.exp(-x1*x3**2 - x10**2*x8 - x11**2*x12 - x16**2*x17 - x20**2*x21 - x24**2*x25 - x28**2*x29 - x34*x36**2 - x38*x39**2 - x43*x45**2 - x47*x48**2 - x5*x6**2 - x52*x54**2 - x55*x56**2 - x60*x62**2 - x63*x64**2 - x68*x70**2 - x71*x72**2)
    x76 = 2*x8
    x77 = x10*x76
    x78 = x16*x17
    x79 = ddq_j1*x14
    x80 = ddq_j2*x77
    x81 = 2*x17
    x82 = x16*x81
    x83 = 2*x21
    x84 = x20*x83
    x85 = x20*x21
    x86 = ddq_j3*x82
    x87 = 2*x25
    x88 = x24*x87
    x89 = x24*x25
    x90 = ddq_j4*x84
    x91 = 2*x29
    x92 = x28*x91
    x93 = x28*x29
    
    K_block_list = []
    K_block_list.append(x75*(-ddq_j1*x12*(-x11**2*x13 + 1) + x14*x19 + x14*x23 + x14*x27 + x14*x31 + x14*x32 + x14*x41 + x14*x50 - x14*x58 - x14*x66 - x14*x74 + x15*x9 + x4 - x7))
    K_block_list.append(x75*(ddq_j1*x15*x8 + x19*x77 + x23*x77 + x27*x77 + x31*x77 + x32*x77 - x37 + x40 + x41*x77 + x50*x77 - x58*x77 - x66*x77 - x74*x77 - x9*(-x10**2*x76 + 1)))
    K_block_list.append(x75*(-x18*(-x16**2*x81 + 1) + x23*x82 + x27*x82 + x31*x82 + x32*x82 + x41*x82 - x46 + x49 + x50*x82 - x58*x82 - x66*x82 - x74*x82 + x78*x79 + x78*x80))
    K_block_list.append(x75*(-x22*(-x20**2*x83 + 1) + x27*x84 + x31*x84 + x32*x84 + x41*x84 + x50*x84 + x57 - x58*x84 - x66*x84 - x74*x84 + x79*x85 + x80*x85 + x85*x86))
    K_block_list.append(x75*(-x26*(-x24**2*x87 + 1) + x31*x88 + x32*x88 + x41*x88 + x50*x88 - x58*x88 + x65 - x66*x88 - x74*x88 + x79*x89 + x80*x89 + x86*x89 + x89*x90))
    K_block_list.append(x75*(ddq_j5*x88*x93 - x30*(-x28**2*x91 + 1) + x32*x92 + x41*x92 + x50*x92 - x58*x92 - x66*x92 + x73 - x74*x92 + x79*x93 + x80*x93 + x86*x93 + x90*x93))
    
    return K_block_list

