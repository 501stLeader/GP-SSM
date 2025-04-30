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
    x12 = sigma_kin_p_2_1_c*x8 + sigma_kin_p_2_1_off + sigma_kin_p_2_1_s*x9
    x13 = x12**2
    x14 = torch.cos(q_j2)
    x15 = sigma_kin_p_2_2_c*torch.cos(q_i2)
    x16 = torch.sin(q_j2)
    x17 = sigma_kin_p_2_2_s*torch.sin(q_i2)
    x18 = sigma_kin_p_2_2_off + x14*x15 + x16*x17
    x19 = x18**2
    x20 = x13*x19
    x21 = dq_i2*sigma_kin_v_2_2
    x22 = dq_i1*sigma_kin_v_2_1
    x23 = x21*x22
    x24 = dq_i1*dq_j1
    x25 = dq_j2*x21
    x26 = sigma_kin_v_2_1*x24 + x25
    x27 = x12*x26**2
    x28 = sigma_kin_p_2_1_c*x4 - sigma_kin_p_2_1_s*x7
    x29 = x19*x28
    x30 = -x14*x17 + x15*x16
    x31 = 2*x18*x26
    x32 = x12*x26
    x33 = x12*x18
    
    K_block_list = []
    K_block_list.append(2*ddq_j1*x0*(sigma_kin_v_2_1**2*x20 + x1*x10**2) + 2*ddq_j2*x20*x23 + 2*dq_j1**2*x0*x11 - 2*dq_j2*x13*x22*x30*x31 - 4*x24*(sigma_kin_v_2_1*x29*x32 + x11*x24) + 2*x27*x29)
    K_block_list.append(2*x33*(ddq_j1*x23*x33 + ddq_j2*dq_i2**2*sigma_kin_v_2_2**2*x33 - dq_j1*x21*x28*x31 - 2*x25*x30*x32 + x27*x30))
    
    return K_block_list

