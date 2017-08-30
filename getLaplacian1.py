import cv2
import numpy as np
import tensorflow as tf
from numpy.linalg import inv
from scipy import sparse
def getLaplacian1(I,consts,win_size = 1,eps = 1e-5):

    neb_size = (win_size*2+1)*(win_size*2+1)
    h,w,c = np.shap(I)[0],np.shape(I)[1],np.shape(I)[2]
    n = h
    m = w

    img_size = w*h
    consts = cv2.erode(consts,-1,element = np.ones(win_size*2+1))

    indsM = np.arange(1,img_size).reshape(h,w,order='F');
    tlen = np.sum(np.sum(consts[win_size+1:win_size,win_size+1:win_size]))*(neb_size*neb_size)

    row_inds = np.zeros(tlen,1)
    col_inds = np.zeros(tlen,1)
    vals = np.zeros(tlen,1)
    len = 0;
    for j in range(1+win_size,w-win_size):
        for i in range(win_size+1,h-win_size):
            if consts == 1:
                continue
            win_inds = indsM[i - win_size : i+win_size,j-win_size:j+win_size]
            win_inds = win_inds[:]
            winI = I[i - win_size:i+win_size,j - win_size:j+win_size]
            winI = winI.reshape(neb_size,c)
            win_mu = np.mean(winI,1)
            win_var = inv(winI*winI/neb_size - win_mu*win_mu+eps)/neb_size

            winI = winI -np.tile(win_mu,neb_size,1)
            tvals = (1+winI*win_var*winI)/neb_size
            row_inds[1+len:neb_size*neb_size+len] = np.reshape(np.tile(win_inds,1,neb_size),neb_size*neb_size,1)
            col_inds[1+len:neb_size*neb_size+len] = np.reshape(np.tile(win_inds,neb_size,1),neb_size*neb_size,1)

            vals[1+len:neb_size**2,len]=tvals[:]

            len = len+neb_size**2

    vals = vals[1:len]
    row_inds = row_inds[1:len]
    col_inds = col_inds[1:len]
    A = sparse.csr_matrix((vals,(row_inds,col_inds)),shape=(img_size,img_size))

    sumA = np.sum(A,1).T.tolist()[0]
    A = sparse.diags([sumA],[0],shape = (img_size,img_size)) - A
    return A
def getLaplacian(img):
    h,w,_ = img.shape
    coo = getLaplacian1(img,np.zeros(shape=(h,w)))
    idx = np.mat([coo.row,coo.col]).transpose()
    return tf.SparseTensor(idx,coo.data,coo.shape)