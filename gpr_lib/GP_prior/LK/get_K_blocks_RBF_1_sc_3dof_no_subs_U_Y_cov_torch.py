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

    
    
    K_block_list = []
    K_block_list.append(0)
    
    return K_block_list

