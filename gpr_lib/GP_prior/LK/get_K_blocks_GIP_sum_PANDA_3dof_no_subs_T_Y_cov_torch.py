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
    x2 = torch.cos(q_i1)
    x3 = torch.sin(q_j1)
    x4 = x2*x3
    x5 = torch.cos(q_j1)
    x6 = torch.sin(q_i1)
    x7 = x5*x6
    x8 = x2*x5
    x9 = x3*x6
    x10 = sigma_kin_p_1_1_c*x8 + sigma_kin_p_1_1_off + sigma_kin_p_1_1_s*x9
    x11 = x1*x10*(sigma_kin_p_1_1_c*x4 - sigma_kin_p_1_1_s*x7)
    x12 = sigma_kin_p_3_1_c*x8 + sigma_kin_p_3_1_off + sigma_kin_p_3_1_s*x9
    x13 = x12**2
    x14 = torch.cos(q_i2)
    x15 = torch.cos(q_j2)
    x16 = x14*x15
    x17 = torch.sin(q_i2)
    x18 = torch.sin(q_j2)
    x19 = x17*x18
    x20 = sigma_kin_p_3_2_c*x16 + sigma_kin_p_3_2_off + sigma_kin_p_3_2_s*x19
    x21 = x20**2
    x22 = torch.cos(q_j3)
    x23 = sigma_kin_p_3_3_c*torch.cos(q_i3)
    x24 = torch.sin(q_j3)
    x25 = sigma_kin_p_3_3_s*torch.sin(q_i3)
    x26 = sigma_kin_p_3_3_off + x22*x23 + x24*x25
    x27 = x26**2
    x28 = x21*x27
    x29 = x13*x28
    x30 = sigma_kin_v_3_1*x29
    x31 = dq_i3*sigma_kin_v_3_3
    x32 = ddq_j3*x31
    x33 = dq_i1*dq_j1
    x34 = dq_i2*dq_j2
    x35 = sigma_kin_v_2_1*x33 + sigma_kin_v_2_2*x34
    x36 = x35**2
    x37 = sigma_kin_p_2_2_c*x16 + sigma_kin_p_2_2_off + sigma_kin_p_2_2_s*x19
    x38 = x37**2
    x39 = sigma_kin_p_2_1_c*x8 + sigma_kin_p_2_1_off + sigma_kin_p_2_1_s*x9
    x40 = x38*x39*(sigma_kin_p_2_1_c*x4 - sigma_kin_p_2_1_s*x7)
    x41 = dq_j3*x31
    x42 = sigma_kin_v_3_1*x33 + sigma_kin_v_3_2*x34 + x41
    x43 = x42**2
    x44 = x12*x43
    x45 = sigma_kin_p_3_1_c*x4 - sigma_kin_p_3_1_s*x7
    x46 = x28*x45
    x47 = sigma_kin_v_3_1*x42
    x48 = -x22*x25 + x23*x24
    x49 = 2*dq_j3*x13*x21*x26*x48
    x50 = ddq_j2*dq_i2
    x51 = x39**2
    x52 = x38*x51
    x53 = dq_i1*(sigma_kin_v_2_1*sigma_kin_v_2_2*x52 + sigma_kin_v_3_2*x30)
    x54 = x14*x18
    x55 = x15*x17
    x56 = x37*x51*(sigma_kin_p_2_2_c*x54 - sigma_kin_p_2_2_s*x55)
    x57 = sigma_kin_v_2_1*x35
    x58 = sigma_kin_p_3_2_c*x54 - sigma_kin_p_3_2_s*x55
    x59 = x13*x20*x27*x58
    x60 = 2*dq_j2
    x61 = x12*x46
    x62 = sigma_kin_v_3_2*x42
    x63 = sigma_kin_v_2_2*x35
    x64 = 2*dq_j1
    x65 = x20*x26
    x66 = x12*x65
    x67 = x31*x66
    x68 = x20*x48
    x69 = x31*x42
    x70 = 2*x12
    
    K_block_list = []
    K_block_list.append(2*ddq_j1*x0*(sigma_kin_v_2_1**2*x52 + sigma_kin_v_3_1**2*x29 + x1*x10**2) + 2*dq_i1*x30*x32 - 2*dq_i1*x47*x49 - 2*dq_i1*x60*(x47*x59 + x56*x57) + 2*dq_j1**2*x0*x11 - 4*x33*(x11*x33 + x40*x57 + x47*x61) + 2*x36*x40 + 2*x44*x46 + 2*x50*x53)
    K_block_list.append(2*ddq_j1*dq_i2*x53 + 2*ddq_j2*dq_i2**2*(sigma_kin_v_2_2**2*x52 + sigma_kin_v_3_2**2*x29) + 2*dq_i2*sigma_kin_v_3_2*x29*x32 - 2*dq_i2*x49*x62 - 2*dq_i2*x64*(x40*x63 + x61*x62) - 4*x34*(x56*x63 + x59*x62) + 2*x36*x56 + 2*x43*x59)
    K_block_list.append(x65*x70*(ddq_j1*dq_i1*sigma_kin_v_3_1*x67 + ddq_j3*dq_i3**2*sigma_kin_v_3_3**2*x66 + sigma_kin_v_3_2*x50*x67 - x12*x26*x58*x60*x69 - x41*x42*x68*x70 + x44*x68 - x45*x64*x65*x69))
    
    return K_block_list

