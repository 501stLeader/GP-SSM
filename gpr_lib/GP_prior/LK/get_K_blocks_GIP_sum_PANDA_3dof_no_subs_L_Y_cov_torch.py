import torch

def get_K_blocks_GIP_sum_PANDA_3dof_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list):

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

    sigma_kin_v_1_1 = sigma_kin_vel_list[0][0]
    sigma_kin_p_1_1_s = sigma_kin_pos_list[0][0]
    sigma_kin_p_1_1_c = sigma_kin_pos_list[0][1]
    sigma_kin_p_1_1_off = sigma_kin_pos_list[0][2]
    sigma_pot_1_s = sigma_pot_list[0][0]
    sigma_pot_1_c = sigma_pot_list[0][1]
    sigma_pot_1_off = sigma_pot_list[0][2]
    sigma_kin_v_2_1 = sigma_kin_vel_list[1][0]
    sigma_kin_p_2_1_s = sigma_kin_pos_list[1][0]
    sigma_kin_p_2_1_c = sigma_kin_pos_list[1][1]
    sigma_kin_p_2_1_off = sigma_kin_pos_list[1][2]
    sigma_kin_v_2_2 = sigma_kin_vel_list[1][1]
    sigma_kin_p_2_2_s = sigma_kin_pos_list[2][0]
    sigma_kin_p_2_2_c = sigma_kin_pos_list[2][1]
    sigma_kin_p_2_2_off = sigma_kin_pos_list[2][2]
    sigma_pot_2_s = sigma_pot_list[1][0]
    sigma_pot_2_c = sigma_pot_list[1][1]
    sigma_pot_2_off = sigma_pot_list[1][2]
    sigma_kin_v_3_1 = sigma_kin_vel_list[2][0]
    sigma_kin_p_3_1_s = sigma_kin_pos_list[3][0]
    sigma_kin_p_3_1_c = sigma_kin_pos_list[3][1]
    sigma_kin_p_3_1_off = sigma_kin_pos_list[3][2]
    sigma_kin_v_3_2 = sigma_kin_vel_list[2][1]
    sigma_kin_p_3_2_s = sigma_kin_pos_list[4][0]
    sigma_kin_p_3_2_c = sigma_kin_pos_list[4][1]
    sigma_kin_p_3_2_off = sigma_kin_pos_list[4][2]
    sigma_kin_v_3_3 = sigma_kin_vel_list[2][2]
    sigma_kin_p_3_3_s = sigma_kin_pos_list[5][0]
    sigma_kin_p_3_3_c = sigma_kin_pos_list[5][1]
    sigma_kin_p_3_3_off = sigma_kin_pos_list[5][2]
    sigma_pot_3_s = sigma_pot_list[2][0]
    sigma_pot_3_c = sigma_pot_list[2][1]
    sigma_pot_3_off = sigma_pot_list[2][2]

    
    x0 = dq_i1**2
    x1 = sigma_kin_v_1_1**2
    x2 = torch.sin(q_j1)
    x3 = torch.cos(q_i1)
    x4 = sigma_kin_p_1_1_c*x3
    x5 = torch.cos(q_j1)
    x6 = torch.sin(q_i1)
    x7 = sigma_kin_p_1_1_s*x6
    x8 = sigma_kin_p_1_1_off + x2*x7 + x4*x5
    x9 = x1*x8*(x2*x4 - x5*x7)
    x10 = sigma_pot_1_c*x3
    x11 = sigma_pot_1_s*x6
    x12 = torch.cos(q_i2)
    x13 = torch.cos(q_j2)
    x14 = x12*x13
    x15 = torch.sin(q_i2)
    x16 = torch.sin(q_j2)
    x17 = x15*x16
    x18 = sigma_pot_2_c*x14 + sigma_pot_2_off + sigma_pot_2_s*x17
    x19 = torch.cos(q_i3)
    x20 = torch.cos(q_j3)
    x21 = x19*x20
    x22 = torch.sin(q_i3)
    x23 = torch.sin(q_j3)
    x24 = x22*x23
    x25 = 1.0*sigma_pot_3_c*x21 + 1.0*sigma_pot_3_off + 1.0*sigma_pot_3_s*x24
    x26 = 2*ddq_j3
    x27 = dq_i3*sigma_kin_v_3_3
    x28 = sigma_kin_p_3_1_c*x3
    x29 = sigma_kin_p_3_1_s*x6
    x30 = sigma_kin_p_3_1_off + x2*x29 + x28*x5
    x31 = x30**2
    x32 = sigma_kin_p_3_2_c*x14 + sigma_kin_p_3_2_off + sigma_kin_p_3_2_s*x17
    x33 = x32**2
    x34 = sigma_kin_p_3_3_c*x21 + sigma_kin_p_3_3_off + sigma_kin_p_3_3_s*x24
    x35 = x34**2
    x36 = x33*x35
    x37 = x31*x36
    x38 = sigma_kin_v_3_1*x37
    x39 = dq_i1*x27*x38
    x40 = sigma_kin_p_2_2_c*x14 + sigma_kin_p_2_2_off + sigma_kin_p_2_2_s*x17
    x41 = x40**2
    x42 = sigma_kin_p_2_1_c*x3
    x43 = sigma_kin_p_2_1_s*x6
    x44 = sigma_kin_p_2_1_off + x2*x43 + x42*x5
    x45 = x41*x44*(x2*x42 - x43*x5)
    x46 = dq_i1*dq_j1
    x47 = dq_i2*dq_j2
    x48 = sigma_kin_v_2_1*x46 + sigma_kin_v_2_2*x47
    x49 = 2*x48**2
    x50 = x30*x36*(x2*x28 - x29*x5)
    x51 = dq_j3*x27
    x52 = sigma_kin_v_3_1*x46 + sigma_kin_v_3_2*x47 + x51
    x53 = 2*x52**2
    x54 = 4*dq_i1
    x55 = sigma_kin_v_3_1*x52
    x56 = x31*x55
    x57 = x19*x23
    x58 = x20*x22
    x59 = x33*x34*(sigma_kin_p_3_3_c*x57 - sigma_kin_p_3_3_s*x58)
    x60 = 2*ddq_j2
    x61 = dq_i2*x60
    x62 = x44**2
    x63 = x41*x62
    x64 = dq_i1*(sigma_kin_v_2_1*sigma_kin_v_2_2*x63 + sigma_kin_v_3_2*x38)
    x65 = 2*ddq_j1
    x66 = sigma_kin_v_2_1*x48
    x67 = x12*x16
    x68 = x13*x15
    x69 = x40*x62*(sigma_kin_p_2_2_c*x67 - sigma_kin_p_2_2_s*x68)
    x70 = x32*x35*(sigma_kin_p_3_2_c*x67 - sigma_kin_p_3_2_s*x68)
    x71 = sigma_pot_1_off + x10*x5 + x11*x2
    x72 = x26*x37
    x73 = sigma_kin_v_3_2*x27
    x74 = x31*x70
    x75 = 4*dq_i2
    x76 = sigma_kin_v_3_2*x52
    x77 = x31*x59
    x78 = sigma_kin_v_2_2*x48
    x79 = 4*x52
    x80 = x27*x79
    
    K_block_list = []
    K_block_list.append(2*dq_j1**2*x0*x9 - dq_j2*x54*(x56*x70 + x66*x69) - dq_j3*x54*x56*x59 + x0*x65*(sigma_kin_v_2_1**2*x63 + sigma_kin_v_3_1**2*x37 + x1*x8**2) + x18*x25*(x10*x2 - x11*x5) + x26*x39 + x45*x49 - 4*x46*(x45*x66 + x46*x9 + x50*x55) + x50*x53 + x61*x64)
    K_block_list.append(dq_i2**2*x60*(sigma_kin_v_2_2**2*x63 + sigma_kin_v_3_2**2*x37) + dq_i2*x64*x65 + dq_i2*x72*x73 - dq_j1*x75*(x45*x78 + x50*x76) - dq_j3*x75*x76*x77 + x25*x71*(sigma_pot_2_c*x67 - sigma_pot_2_s*x68) - 4*x47*(x69*x78 + x74*x76) + x49*x69 + x53*x74)
    K_block_list.append(dq_i3**2*sigma_kin_v_3_3**2*x72 - dq_j1*x50*x80 - dq_j2*x74*x80 + 1.0*x18*x71*(sigma_pot_3_c*x57 - sigma_pot_3_s*x58) + x37*x61*x73 + x39*x65 - x51*x77*x79 + x53*x77)
    
    return K_block_list

