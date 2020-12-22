import numpy as np
import h5py
import cv2
import os 
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True,
                    metavar="/path/to/dataset/",
                    help='Root directory of the dataset')
args = parser.parse_args()
ch = args.path 
os.chdir(ch)
l=glob.glob('track_*.h5')
ratio=0.7
summary=[]
log=[]
path = 'updated'
os.mkdir(ch+path)
kernel=np.ones((3,3))
no_det_val=2
imp_ratio=ratio
uncertainty_threshold=0.51

#interpolation
def interp_shape(top,bottom):
    roi_mask_0,minimum_0,maximum_0,_=compute_roi(top,True)
    roi_mask_1,minimum_1,maximum_1,_=compute_roi(bottom,True) 
    minimum=(int((minimum_0[0]+minimum_1[0])/2),int((minimum_0[1]+minimum_1[1])/2))
    maximum=(int((maximum_0[0]+maximum_1[0])/2),int((maximum_0[1]+maximum_1[1])/2))
    size=(maximum[1]-minimum[1]+1,maximum[0]-minimum[0]+1)
    top_u=cv2.resize(roi_mask_0.astype(np.uint8), dsize=size,interpolation=cv2.INTER_NEAREST)
    bottom_u=cv2.resize(roi_mask_1.astype(np.uint8), dsize=size,interpolation=cv2.INTER_NEAREST)
    #inter=top_u*bottom_u
    inter=top_u+bottom_u
    inter[inter>0]=1
    tmp=np.zeros_like(top)
    tmp[minimum[0]:maximum[0]+1,minimum[1]:maximum[1]+1]=inter
    return tmp

#propagation
def compute_roi(ref_im,flag):
    if flag:
        minimum=(np.min(np.where(ref_im==1)[0]),np.min(np.where(ref_im==1)[1]))
        maximum=(np.max(np.where(ref_im==1)[0]),np.max(np.where(ref_im==1)[1]))
    else:
        minimum=(np.min(np.where(ref_im>0)[0]),np.min(np.where(ref_im>0)[1]))
        maximum=(np.max(np.where(ref_im>0)[0]),np.max(np.where(ref_im>0)[1]))        
    midpoint=(int((minimum[0]+maximum[0])/2),int((minimum[1]+maximum[1])/2))
    roi_mask=np.zeros((maximum[0]-minimum[0]+1,maximum[1]-minimum[1]+1))
    roi_mask=ref_im[minimum[0]:maximum[0]+1, minimum[1]:maximum[1]+1]
    return roi_mask,minimum,maximum,midpoint

def hallucinate(ref_im, bet_im, entropy):
    mask=np.zeros_like(ref_im)
    mask[:]=ref_im
    mask[mask==1]=2
    if np.sum(ref_im==1)>0:
        flag=True
    else:
        flag=False
    ref_mask,ref_min,ref_max, ref_mid =compute_roi(ref_im,flag)
    bet_mask,bet_min,bet_max, bet_mid =compute_roi(bet_im,flag)
    size=list(ref_mask.shape)
    if (size[0])%2==0:
        size[0]+=1
    if (size[1])%2==0:
        size[1]+=1
    size=size[::-1]
    size=tuple(size)
    scaled_mask = cv2.resize(bet_mask, dsize=size,interpolation=cv2.INTER_NEAREST)
    scaled_mask[scaled_mask==2]=0
    min_0=0
    shift_0_0=0
    max_0=ref_im.shape[0]
    shift_0_1=0
    min_1=0
    shift_1_0=0
    max_1=ref_im.shape[1]
    shift_1_1=0
    if int(ref_mid[0]-((size[1]-1)/2))>0:
        min_0=int(ref_mid[0]-((size[1]-1)/2))
    else:
        shift_0_0=-int(ref_mid[0]-((size[1]-1)/2))
    if int(ref_mid[1]-((size[0]-1)/2))>0:
        min_1=int(ref_mid[1]-((size[0]-1)/2))
    else:
        shift_1_0=-int(ref_mid[1]-((size[0]-1)/2))    
    if int(ref_mid[0]+1+((size[1]-1)/2))<max_0:
        max_0=int(ref_mid[0]+1+((size[1]-1)/2))
    else:
        shift_0_1=max_0-int(ref_mid[0]+1+((size[1]-1)/2))        
    if int(ref_mid[1]+1+((size[0]-1)/2))<max_1:
        max_1=int(ref_mid[1]+1+((size[0]-1)/2))
    else:
        shift_1_1=max_1-int(ref_mid[1]+1+((size[0]-1)/2)) 
    mask[min_0:max_0,min_1:max_1]+=10*scaled_mask[shift_0_0:scaled_mask.shape[0]-shift_0_1,shift_1_0:scaled_mask.shape[1]-shift_1_1]
    mask[mask>9]=1
    return mask

#tracking
for f in l:    
    ref = f.split('_')[1]
    track=h5py.File(ch+f,'r')['track'][()]
    entropy=h5py.File(ch+'/entropy_'+ref,'r')['entropy'][()]
    # compute uncertainty 
    uncertainty_dic = {}
    ag_uncertainty=[]
    for i,j in enumerate(track):
        uncertainty_dic[i] = []
        ret, labels = cv2.connectedComponents((j==1).astype(np.uint8))
        for k in range(ret):
            if np.unique(j[labels==k])[0]==1.0:
                uncertainty=np.mean(np.extract(cv2.erode((labels==k).astype(np.uint8),kernel,iterations=0),entropy[i]))
                if np.isnan(uncertainty):
                    uncertainty=no_det_val
                ag_uncertainty.append((uncertainty,i))
                uncertainty_dic[i].append((uncertainty,i))
    for k in uncertainty_dic:
        if len(uncertainty_dic[k])==0:
            uncertainty_dic[k].append(no_det_val)
            ag_uncertainty.append((no_det_val,k))
    ag_uncertainty=[(x[1],x[0]) for x in sorted(ag_uncertainty)]  
    no_det_occ = np.sum([1 for x in ag_uncertainty if x[1]==no_det_val])
    if no_det_occ>0:
        no_det_idx = [x[0] for x in ag_uncertainty[-no_det_occ:]]
        no_det_idx = [[el] for el in np.sort(no_det_idx)]
        for i in range(len(no_det_idx)-1,0,-1):
            if no_det_idx[i][0]==no_det_idx[i-1][0]+1:
                no_det_idx[i-1]=no_det_idx[i-1]+no_det_idx[i]
                no_det_idx.remove(no_det_idx[i])
        for pos,k in enumerate(no_det_idx):
            if k[0]==0:
                k=k[::-1]
                no_det_idx[pos]=k
                continue
            elif k[-1]==len(ag_uncertainty)-1:
                continue
            else:
                tmp=[]
                while len(k)>0:
                    tmp.append(k.pop(0))
                    if len(k)>0:
                        tmp.append(k.pop(-1))
                no_det_idx[pos]=tmp  
        no_det_idx = [item for sublist in no_det_idx for item in sublist]
        ag_uncertainty[-no_det_occ:]=[(i,no_det_val) for i in no_det_idx] 
    for i in ag_uncertainty:
        masks=np.zeros((3,track.shape[1],track.shape[2]))
        masks[0][:]=track[i[0]]
        masks[0][masks[0]>0]=2
        ref=np.zeros((track.shape[1],track.shape[2]))
        ref[:]=track[i[0]]
        nuc_slice=np.zeros((track.shape[1],track.shape[2]))
        if len(uncertainty_dic[i[0]])==1:
            masks[0][:]=track[i[0]]
            nuc_slice[track[i[0]]==1]=1
        else:
            ret, labels = cv2.connectedComponents((track[i[0]]==1).astype(np.uint8))
            for k in range(ret):
                if np.unique(track[i[0]][labels==k])[0]==1.0:
                    uncertainty=np.mean(np.extract(cv2.erode((labels==k).astype(np.uint8),kernel,iterations=0),entropy[i[0]]))
                    if np.isnan(uncertainty):
                        uncertainty=no_det_val
                    if uncertainty == i[1]:
                        nuc_slice[labels==k]=1
                        break
            masks[0]=masks[0]+nuc_slice
            masks[0][masks[0]%2==1]=1
        ref+=nuc_slice 
        if i[0] == 0:
            ret, labels = cv2.connectedComponents((track[i[0]+1]==1).astype(np.uint8))
            cands = np.zeros((ret,track.shape[1],track.shape[2]))
            count=0
            for k in range(ret):
                if np.unique(track[i[0]+1][labels==k])[0]==1.0:
                    cands[count][labels==k]=1
                    count+=1
            agg=np.sum(np.sum(cands*nuc_slice,axis=1),axis=1)
            index=np.argmax(agg)
            if agg[index]>0 :
                masks[2][:]=track[i[0]+1]
                masks[2][masks[2]>0]=2
                masks[2]+=cands[index]
                masks[2][masks[2]%2==1]=1
            else:
                masks[2][:]=track[i[0]+1]
                masks[2][masks[2]>0]=2                    
            if i[1]==no_det_val:
                masks[2][:]=track[i[0]+1]
        elif i[0] == track.shape[0]-1:
            ret, labels = cv2.connectedComponents((track[i[0]-1]==1).astype(np.uint8))
            cands = np.zeros((ret,track.shape[1],track.shape[2]))
            count=0
            for k in range(ret):
                if np.unique(track[i[0]-1][labels==k])[0]==1.0:
                    cands[count][labels==k]=1
                    count+=1
            agg=np.sum(np.sum(cands*nuc_slice,axis=1),axis=1)
            index=np.argmax(agg)
            if agg[index]>0:
                masks[1][:]=track[i[0]-1]
                masks[1][masks[1]>0]=2
                masks[1]+=cands[index]
                masks[1][masks[1]%2==1]=1
            else:
                masks[1][:]=track[i[0]-1]
                masks[1][masks[1]>0]=2  
            if i[1]==no_det_val:
                masks[1][:]=track[i[0]-1]
        else:
            ret, labels = cv2.connectedComponents((track[i[0]+1]==1).astype(np.uint8))
            cands = np.zeros((ret,track.shape[1],track.shape[2]))
            count=0
            for k in range(ret):
                if np.unique(track[i[0]+1][labels==k])[0]==1.0:
                    cands[count][labels==k]=1
                    count+=1
            agg=np.sum(np.sum(cands*nuc_slice,axis=1),axis=1)
            index=np.argmax(agg)
            if agg[index]>0:
                masks[2][:]=track[i[0]+1]
                masks[2][masks[2]>0]=2
                masks[2]+=cands[index]
                masks[2][masks[2]%2==1]=1
            else:
                masks[2][:]=track[i[0]+1]
                masks[2][masks[2]>0]=2  
            if i[1]==no_det_val:
                masks[2][:]=track[i[0]+1]
            ret, labels = cv2.connectedComponents((track[i[0]-1]==1).astype(np.uint8))                
            cands = np.zeros((ret,track.shape[1],track.shape[2]))
            count=0
            for k in range(ret):
                if np.unique(track[i[0]-1][labels==k])[0]==1.0:
                    cands[count][labels==k]=1
                    count+=1
            agg=np.sum(np.sum(cands*nuc_slice,axis=1),axis=1)
            index=np.argmax(agg)
            if agg[index]>0:
                masks[1][:]=track[i[0]-1]
                masks[1][masks[1]>0]=2
                masks[1]+=cands[index]
                masks[1][masks[1]%2==1]=1
            else:
                masks[1][:]=track[i[0]-1]
                masks[1][masks[1]>0]=2  
            if i[1]==no_det_val:
                masks[1][:]=track[i[0]-1]
        neighbors=np.zeros((3))
        neighbors[0]=i[1]
        if not(1.0 in list(np.unique(masks[1]))):
            neighbors[1]=no_det_val
        else:
            neighbors[1]=np.mean(np.extract(cv2.erode((masks[1]==1).astype(np.uint8),kernel,iterations=0),entropy[i[0]-1]))
        if not(1.0 in list(np.unique(masks[2]))):
            neighbors[2]=no_det_val
        else:
            neighbors[2]=np.mean(np.extract(cv2.erode((masks[2]==1).astype(np.uint8),kernel,iterations=0),entropy[i[0]+1])) 
        if neighbors[2]<((1+imp_ratio)/2)*neighbors[0]>neighbors[1] and neighbors[0]>uncertainty_threshold:
            image_1=(masks[1]==1)
            image_2=(masks[2]==1)
            mask=interp_shape(image_1,image_2)
            ref = ref+10*mask
            ref[ref>9]=1 
            track[i[0]]=ref
        elif neighbors[1]<imp_ratio*neighbors[0] and neighbors[0]>uncertainty_threshold:
            mask=hallucinate(masks[0],masks[1],entropy[i[0]-1])
            mask=mask%2
            ref = ref+10*mask
            ref[ref>9]=1 
            track[i[0]]=ref               
        elif imp_ratio*neighbors[0]>neighbors[2] and neighbors[0]>uncertainty_threshold:
            mask=hallucinate(masks[0],masks[2],entropy[i[0]+1])
            mask=mask%2
            ref = ref+10*mask
            ref[ref>9]=1 
            track[i[0]]=ref
    output=h5py.File(ch+path+'/prop_'+f,'w')
    output['track']=track
    output.close()





