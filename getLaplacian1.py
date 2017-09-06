import cv2
import numpy as np
import tensorflow as tf
from numpy.linalg import inv
from scipy import sparse
def getLaplacian1(I,consts,win_size = 1,eps = 1e-5):

    neb_size = (win_size*2+1)*(win_size*2+1)
    h,w,c = I.shape
    n = h
    m = w

    img_size = w*h
    consts = cv2.erode(consts,-1,element = np.ones(win_size*2+1))

    indsM = np.arange(1,img_size).reshape(h,w,order='F');
    tlen = int((-consts[win_size+1:win_size,win_size+1:win_size]+1).sum()*(neb_size**2))

    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    len = 0;
    for j in range(1+win_size,w-win_size):
        for i in range(win_size+1,h-win_size):
            if consts[i,j] == 1:
                continue
            win_inds = indsM[i - win_size : i+win_size,j-win_size:j+win_size]
            win_inds = win_inds.ravel(order='F')
            winI = I[i - win_size:i+win_size,j - win_size:j+win_size]
            winI = winI.reshape(neb_size, c)
            win_mu = np.mean(winI,axis=0).reshape(c,1)
            win_var = np.linalg.inv(winI*winI/neb_size - win_mu*win_mu.T+eps/neb_size*np.identity(c))

            winII = winI -np.repeat(win_mu.transpose(),neb_size,0)
            tvals = (1+winII*win_var*winII.T)/neb_size

            ind_mat = np.broadcast_to(win_inds,(neb_size,neb_size))

            row_inds[len:neb_size**2+len] = ind_mat.ravel(order='C')
            col_inds[len:neb_size**2+len] = ind_mat.ravel(order='F')
            vals[1+len:neb_size**2,len]=tvals.ravel(order='F')

            len = len+neb_size**2

    vals = vals.ravel(order='F')[0:len]
    row_inds = row_inds.ravel(order='F')[0:len]
    col_inds = col_inds.ravel(order='F')[0:len]
    A = sparse.csr_matrix((vals,(row_inds,col_inds)),shape=(img_size,img_size))

    sumA = np.sum(A,1).T.tolist()[0]
    A = sparse.diags([sumA],[0],shape = (img_size,img_size)) - A
    return A
def getLaplacian(img):
    h,w,_ = img.shape
    coo = getLaplacian1(img,np.zeros(shape=(h,w))).tocoo()
    idx = np.mat([coo.row,coo.col]).transpose()
    return tf.SparseTensor(idx,coo.data,coo.shape)