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

    
    x0 = torch.cos(q_j1)
    x1 = sigma_kin_p_1_1_c*torch.cos(q_i1)
    x2 = torch.sin(q_j1)
    x3 = sigma_kin_p_1_1_s*torch.sin(q_i1)
    x4 = sigma_kin_p_1_1_off + x0*x1 + x2*x3
    
    K_block_list = []
    K_block_list.append(2*dq_i1**2*sigma_kin_v_1_1**2*x4*(ddq_j1*x4 - dq_j1**2*(-x0*x3 + x1*x2)))
    
    return K_block_list

