import numpy as np 
from numpy import inf

def norm_(samples,gt_sample):

    output = np.zeros((samples.shape[0],samples.shape[1],samples.shape[2],samples.shape[-1])).astype(np.float32).astype(float)

    for idx,sample in enumerate (samples):
        output[idx,:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        output[idx,:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        output[idx,:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))

    gt_output = np.zeros((samples.shape[0],samples.shape[1],samples.shape[2],samples.shape[-1])).astype(np.float32).astype(float)
    for idx,sample in enumerate (gt_sample):
        gt_output[idx,:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        gt_output[idx,:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
        gt_output[idx,:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))


    #output = output[np.logical_not(np.isnan(x))]
    #gt_output = gt_output[np.logical_not(np.isnan(x))]
    output[output == inf] = 0
    gt_output[gt_output == inf] = 0

    tmp = np.multiply(output,gt_output)
    tmp = 1.0 - np.sum(tmp,axis=3,dtype=np.float32)
    
    #tmp = np.arccos(tmp)
    return  np.sum(tmp)/samples.shape[0]
    #return np.asarray(output)


