import numpy as np
import scipy.misc

def compute_ei(normal_map):
    Ei_cum_total =0.0
    min_size =3
    for idx in range(normal_map.shape[0]):
        normal_map1 = np.squeeze(normal_map[idx,:,:,:])
        mask = (normal_map1[:,:,0] >-1.).astype(int)
        mask_size= mask.shape
        mask = np.reshape(mask,-1)
        normal_map1 = np.reshape(normal_map1,(-1,3))
        tmp1 = normal_map1[:,0]
        normal1 = tmp1[mask >0]
        tmp2 = normal_map1[:,1]
        normal2 = tmp2[mask >0]
        tmp3 = normal_map1[:,2]
        normal3 = tmp1[mask >0]
        normal = np.transpose(np.squeeze(np.dstack((normal1,normal2,normal3)))) 
        if min_size > normal.shape[-1]:
            continue
        #print('normal shape:',normal.shape)
        #normal = normal_map[mask>0]
        #normal= np.transpose(np.reshape(normal,(-1,3)))
        #normal = normal/127.5-1

        grid = mask
        cum_grid = np.zeros((grid.shape[0])).astype(int)
        cum_grid[grid ==1] = range(1, np.sum(grid)+1)
        cum_grid = np.reshape(cum_grid,(mask_size[0],mask_size[1]))
 
        dw=2
        overlap=1
        start= dw
        step= 2*dw+overlap+1
        end=mask_size[0]-dw
        Ei_cum = 0.0
        for j in range(start,end,step):
            for i in range(start,end,step):
              
                pcgrid = cum_grid[j-dw:j+dw+1,i-dw:i+dw+1]
                bpcgrid = (pcgrid>0).astype(int)
                pidx = pcgrid[pcgrid>0] 
                #print(np.max(pidx))
                dgrid = np.zeros((2*dw+1, 2*dw+1))
                sz_dgrid = dgrid.shape
                dgrid[bpcgrid ==1] =range(1,np.sum(bpcgrid)+1)
                h = sz_dgrid[0]-2
                w = sz_dgrid[1]-2
                didx =[]
                didx.append(np.reshape(dgrid[1:-1, 1:-1],h*w,1))
                didx.append(np.reshape(dgrid[0:-2, 1:-1],h*w,1)) #y-1
                didx.append(np.reshape(dgrid[1:-1, 0:-2],h*w,1)) #x-1
                didx.append(np.reshape(dgrid[2:, 1:-1],h*w,1)) #y+1
                didx.append(np.reshape(dgrid[1:-1, 2:],h*w,1)) #x+1
                didx= np.asarray(didx).astype(int)
                didx = np.transpose(didx)
                iszero = (didx == 0).astype(int)
                dvalid=np.sum(iszero, 1)
                didx=didx[dvalid==0]
                if pidx.shape[0] != 0:
                #if n_vec.shape[1] !=0:
                    n_vec = normal.take(pidx,axis=1)
                    pq0_=n_vec[0:2,:]/np.tile(n_vec[2,:],(2,1))
                    tmp = pq0_.take(didx[:,3])
                    tmp2 = pq0_.take(didx[:,1])
                    tmp3 = pq0_.take(didx[:,4])
                    tmp4 = pq0_.take(didx[:,2])
                    Ei = 0.5*(tmp - tmp2 - tmp3 + tmp4)
                    Ei_cum = Ei_cum + np.sum(Ei).astype(float)
        Ei_cum_total += np.abs(Ei_cum)
    return Ei_cum_total/normal_map.shape[0]

