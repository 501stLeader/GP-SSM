import torch

def get_K_blocks_GIP_sum_InversePendulum_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list):

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

    
    x0 = torch.cos(q_i1)
    x1 = torch.sin(q_j1)
    x2 = x0*x1
    x3 = torch.cos(q_j1)
    x4 = torch.sin(q_i1)
    x5 = x3*x4
    x6 = torch.cos(q_i2)
    x7 = torch.cos(q_j2)
    x8 = x6*x7
    x9 = torch.sin(q_i2)
    x10 = torch.sin(q_j2)
    x11 = x10*x9
    x12 = 2*dq_i1**2
    x13 = sigma_kin_v_1_1**2
    x14 = x0*x3
    x15 = x1*x4
    x16 = sigma_kin_p_1_1_c*x14 + sigma_kin_p_1_1_off + sigma_kin_p_1_1_s*x15
    x17 = x13*x16*(sigma_kin_p_1_1_c*x2 - sigma_kin_p_1_1_s*x5)
    x18 = sigma_kin_p_2_1_c*x14 + sigma_kin_p_2_1_off + sigma_kin_p_2_1_s*x15
    x19 = x18**2
    x20 = sigma_kin_p_2_2_c*x8 + sigma_kin_p_2_2_off + sigma_kin_p_2_2_s*x11
    x21 = x20**2
    x22 = x19*x21
    x23 = 2*x22
    x24 = ddq_j2*x23
    x25 = dq_i2*sigma_kin_v_2_2
    x26 = dq_i1*sigma_kin_v_2_1*x25
    x27 = x18*x21*(sigma_kin_p_2_1_c*x2 - sigma_kin_p_2_1_s*x5)
    x28 = dq_i1*dq_j1
    x29 = dq_j2*x25
    x30 = sigma_kin_v_2_1*x28 + x29
    x31 = 2*x30**2
    x32 = sigma_kin_v_2_1*x30
    x33 = x10*x6
    x34 = x7*x9
    x35 = x19*x20*(sigma_kin_p_2_2_c*x33 - sigma_kin_p_2_2_s*x34)
    x36 = 4*x35
    
    K_block_list = []
    K_block_list.append(ddq_j1*x12*(sigma_kin_v_2_1**2*x22 + x13*x16**2) - dq_i1*dq_j2*x32*x36 + dq_j1**2*x12*x17 + x24*x26 + x27*x31 - 4*x28*(x17*x28 + x27*x32) + 1.0*(sigma_pot_1_c*x2 - sigma_pot_1_s*x5)*(sigma_pot_2_c*x8 + sigma_pot_2_off + sigma_pot_2_s*x11))
    K_block_list.append(ddq_j1*x23*x26 + dq_i2**2*sigma_kin_v_2_2**2*x24 - 4*dq_j1*x25*x27*x30 - x29*x30*x36 + x31*x35 + 1.0*(sigma_pot_2_c*x33 - sigma_pot_2_s*x34)*(sigma_pot_1_c*x14 + sigma_pot_1_off + sigma_pot_1_s*x15))
    
    return K_block_list

