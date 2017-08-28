import cv2
import numpy as np

def boxfilter(I,rad):
    res = np.zeros(np.shape(I))
    cv2.boxfilter(I,res,rad)
    return res
def guider_filter_color(I,P,rad,eps):

    N = boxfilter(np.ones(np.shape(I)),rad)

    hgt = np.shape(I)[0]
    wid = np.shape(I)[1]
    mean_I_r = boxfilter(I[::1],rad)/N
    mean_I_g = boxfilter(I[::2],rad)/N
    mean_I_b = boxfilter(I[::3],rad)/N

    mean_p = boxfilter(p,rad)/N

    mean_IP_r = boxfilter(I[::1],rad)/N
    mean_IP_g = boxfilter(I[::2],rad)/N
    mean_IP_b = boxfilter(I[::3],rad)/N

    cov_IP_r = mean_IP_r - mean_I_r *mean_p
    cov_IP_g = mean_IP_g - mean_I_g *mean_p
    cov_IP_b = mean_IP_b - mean_I_b *mean_p

    var_I_rr = boxfilter(I[::1]*I[::1],rad)/N - mean_I_r * mean_I_r
    var_I_rg = boxfilter(I[::1]*I[::2],rad)/N - mean_I_r * mean_I_g
    var_I_rb = boxfilter(I[::1]*I[::3],rad)/N - mean_I_r * mean_I_b

    var_I_gg = boxfilter(I[::2]*I[::2],rad)/N - mean_I_g * mean_I_g
    var_I_gb = boxfilter(I[::2]*I[::3],rad)/N - mean_I_g * mean_I_b
    var_I_bb = boxfilter(I[::3]*I[::3],rad)/N - mean_I_b * mean_I_b

    a = np.ones(np.shape(I))

    for y in range(1,hgt):
        for x in range(1,wid):
            sigma = [var_I_rr[y:x],var_I_rg[y:x],var_I_rb[y:x],
            var_I_rg[y:x],var_I_gg[y:x],var_I_gb[y:x],
                     var_I_rb[y:x],var_I_gb[y:x],var_I_bb[y:x]]
            cov_IP = [cov_IP_r[y:x],cov_IP_g[y:x],cov_IP_b[y:x]]
            
            a[y:x:] = cov_IP*np.invert(sigma+eps*np.eye(3))

    b = mean_p - a[::1]*mean_I_r - a[::2]*mean_I_g - a[::3] * mean_I_b

    q = (boxfilter(a[::1],rad)*I[::1]+boxfilter(a[::2],rad)*I[::2]+boxfilter(a[::3],rad)*I[::3]+boxfilter(b,rad))/N
    return q