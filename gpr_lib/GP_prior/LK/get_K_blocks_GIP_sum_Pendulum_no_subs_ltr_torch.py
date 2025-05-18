import torch

def get_K_blocks_GIP_sum_Pendulum_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list):

    q_i1 = X1[:,pos_indices[0]:pos_indices[0]+1]
    dq_i1 = X1[:,vel_indices[0]:vel_indices[0]+1]
    ddq_i1 = X1[:,acc_indices[0]:acc_indices[0]+1]
    q_j1 = X2[:,pos_indices[0]:pos_indices[0]+1].transpose(1,0)
    dq_j1 = X2[:,vel_indices[0]:vel_indices[0]+1].transpose(1,0)
    ddq_j1 = X2[:,acc_indices[0]:acc_indices[0]+1].transpose(1,0)

    sigma_kin_v_1_1 = sigma_kin_vel_list[0][0]
    sigma_kin_p_1_1_s = sigma_kin_pos_list[0][0]
    sigma_kin_p_1_1_c = sigma_kin_pos_list[0][1]
    sigma_kin_p_1_1_off = sigma_kin_pos_list[0][2]
    sigma_pot_1_s = sigma_pot_list[0][0]
    sigma_pot_1_c = sigma_pot_list[0][1]
    sigma_pot_1_off = sigma_pot_list[0][2]

    
    x0 = torch.sin(q_j1)
    x1 = torch.sin(q_i1)
    x2 = x0*x1
    x3 = torch.cos(q_i1)
    x4 = torch.cos(q_j1)
    x5 = x3*x4
    x6 = x0*x3
    x7 = x1*x4
    x8 = sigma_kin_p_1_1_c*x6 - sigma_kin_p_1_1_s*x7
    x9 = sigma_kin_p_1_1_c*x5 + sigma_kin_p_1_1_off + sigma_kin_p_1_1_s*x2
    x10 = ddq_i1*x9
    x11 = x10*x8
    x12 = dq_j1**2
    x13 = sigma_kin_v_1_1**2
    x14 = 4*x13
    x15 = x12*x14
    x16 = dq_i1**2
    x17 = x16*x9*(sigma_kin_p_1_1_c*x2 + sigma_kin_p_1_1_s*x5)
    x18 = 2*x12*x13
    x19 = x16*(sigma_kin_p_1_1_c*x7 - sigma_kin_p_1_1_s*x6)
    x20 = x19*x8
    
    K_block_list = []
    K_block_list.append(ddq_j1*x14*x9*(x10 - x19) + 1.0*sigma_pot_1_c*x2 + 1.0*sigma_pot_1_s*x5 + x11*x15 + x15*(-2*x11 + x17 + x20) - x17*x18 - x18*x20)
    
    return K_block_list

