import numpy as np
import scipy.misc

normal_map = scipy.misc.imread('./normal_map_1.bmp').astype(float)
#normal_map =normal_map[333:480,260:356,:]

#making mask
mask = (np.squeeze(normal_map[:,:,0]) >0).astype(int)
sum = np.sum(mask)
mask_size= mask.shape

normal = normal_map[mask>0]
normal= np.transpose(np.reshape(normal,(-1,3)))
normal = normal/127.5-1

grid = mask
cum_grid = np.zeros((grid.shape[0],grid.shape[0])).astype(int)
cum_grid[grid ==1] = range(1, np.sum(grid)+1)

dw=5
overlap=3
start= dw
step= 2*dw+overlap+1
end=mask_size[0]-dw
Ei_cum =0.0

for j in range(start,end,step):
    for i in range(start,end,step):
        print('i %s j %s' % (i,j))
        pcgrid = cum_grid[j-dw:j+dw+1,i-dw:i+dw+1]
        bpcgrid = (pcgrid>0).astype(int)
        pidx = pcgrid[pcgrid>0]
        dgrid = np.zeros((2*dw+1, 2*dw+1))
        sz_dgrid = dgrid.shape
        dgrid[bpcgrid ==1] =range(1,np.sum(bpcgrid)+1)

        h = sz_dgrid[0]-2
        w = sz_dgrid[1]-2
        didx =[]
        didx.append(np.reshape(dgrid[1:-1, 1:-1],h*w,1))
        didx.append(np.reshape(dgrid[0:-2, 1:-1],-1)) #y-1
        didx.append(np.reshape(dgrid[1:-1, 0:-2],-1)) #x-1
        didx.append(np.reshape(dgrid[2:, 1:-1], -1)) #y+1
        didx.append(np.reshape(dgrid[1:-1, 2:], -1)) #x+1
        didx= np.asarray(didx).astype(int)
        didx = np.transpose(didx)

        iszero = (didx == 0).astype(int)
        dvalid=np.sum(iszero, 1)
        didx=didx[dvalid==0]
        n_vec = normal.take(pidx,axis=1)

        if n_vec.shape[1] !=0:

            pq0_=n_vec[0:2,:]/np.tile(n_vec[2,:],(2,1))
            tmp = pq0_.take(didx[:,3])
            tmp2 = pq0_.take(didx[:,1])
            tmp3 = pq0_.take(didx[:,4])
            tmp4 = pq0_.take(didx[:,2])
            Ei = 0.5*(tmp - tmp2 - tmp3 + tmp4)
            Ei_cum = Ei_cum + np.sum(Ei)

print('Ei cum:',Ei_cum)

