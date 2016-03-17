import numpy as np
import scipy.misc
import os
def compute_ei(normal_map):
    Ei_cum_total =0.0
    min_size =3
    ch = normal_map.shape[-1]
    
    output = np.zeros((normal_map.shape[0],normal_map.shape[1],normal_map.shape[2],normal_map.shape[-1])).astype(np.float32).astype(float)
    for idx,sample in enumerate (normal_map):
        output[idx,:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        output[idx,:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        output[idx,:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))


    for idx in range(output.shape[0]):
        normal_map1 = np.squeeze(output[idx,:,:,:])
        mask = (normal_map1[:,:,0] >-1.).astype(int)
        mask_size= mask.shape
        mask = np.reshape(mask,-1,order='F')
        normal_map1 = np.reshape(normal_map1,(-1,ch),order='F')
        tmp1 = normal_map1[:,0]
        normal1 = tmp1[mask == 1]
        tmp2 = normal_map1[:,1]
        normal2 = tmp2[mask == 1]
        tmp3 = normal_map1[:,2]
        normal3 = tmp3[mask == 1]
        normal = np.transpose(np.squeeze(np.dstack((normal1,normal2,normal3)))) 
        if min_size > normal.shape[-1]:
            continue

        grid = mask
        cum_grid = np.zeros((grid.shape[0])).astype(int) -1
        cum_grid[grid ==1] = range(0, np.sum(grid))
        cum_grid = np.reshape(cum_grid,(mask_size[0],mask_size[1]),order='F')
 
        dw=5
        overlap=3
        start= dw
        step= 2*dw+overlap
        end1 = mask_size[0]-dw
        end2 = mask_size[1]-dw
        Ei_cum = 0.0
        for j in range(start,end1,step):
            for i in range(start,end2,step):
              
                pcgrid = cum_grid[j-dw:j+dw+1,i-dw:i+dw+1]
                pcgrid = np.reshape(pcgrid,-1,order='F')
                bpcgrid = (pcgrid>-1).astype(int)
                pidx = pcgrid[pcgrid>-1] 
                
                dgrid = np.zeros((2*dw+1)* (2*dw+1)) -1
                dgrid[bpcgrid ==1] =range(0,np.sum(bpcgrid))
                dgrid = np.reshape(dgrid,(2*dw+1,2*dw+1),order ='F')
                sz_dgrid = dgrid.shape
                #h = sz_dgrid[0]-2
                #w = sz_dgrid[1]-2
                didx =[]
                didx.append(np.reshape(dgrid[1:-1, 1:-1],-1,order='F'))
                didx.append(np.reshape(dgrid[0:-2, 1:-1],-1,order='F')) #y-1
                didx.append(np.reshape(dgrid[1:-1, 0:-2],-1,order='F')) #x-1
                didx.append(np.reshape(dgrid[2:, 1:-1],-1,order='F')) #y+1
                didx.append(np.reshape(dgrid[1:-1, 2:],-1,order='F')) #x+1
                didx = np.asarray(didx).astype(int)
                didx = np.transpose(didx)
                didx = didx[np.prod(didx+1,axis=1)!=0]
                """
                iszero = (didx == 0).astype(int)
                dvalid=np.sum(iszero, 1)
             
                tmp = didx[:,0]
                didx1 = tmp[dvalid == 0]
                tmp = didx[:,1]
                didx2 = tmp[dvalid == 0]
                tmp = didx[:,2]
                didx3 = tmp[dvalid == 0]
                tmp = didx[:,3]
                didx4 = tmp[dvalid == 0]
                tmp = didx[:,4]
                didx5 = tmp[dvalid == 0]
                didx =np.dstack((didx1, didx2, didx3, didx4, didx5))
                didx = np.reshape(didx,(didx1.shape[0],5),order='F')
                """
                n_vec = normal.take(pidx,axis=1)

                if n_vec.shape[1] != 0:
                    n_vec = normal.take(pidx,axis=1)
                    pq0_=n_vec[0:2,:]/np.tile(n_vec[2,:],(2,1))
                    tmp = pq0_.take(didx[:,3])
                    tmp2 = pq0_.take(didx[:,1])
                    tmp3 = pq0_.take(didx[:,4])
                    tmp4 = pq0_.take(didx[:,2])
                    Ei = 0.5*(tmp - tmp2 - tmp3 + tmp4)
                    Ei_cum = Ei_cum + np.sum(Ei).astype(float)
        Ei_cum_total += np.abs(Ei_cum)
    return Ei_cum_total

