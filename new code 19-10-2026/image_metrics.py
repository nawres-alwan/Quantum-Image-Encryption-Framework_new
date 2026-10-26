"""Explicit image metrics; inspection previews are not security certificates."""
import math
import numpy as np


def entropy_channels(image):
    result=[]
    for c in range(3):
        counts=np.bincount(image[:,:,c].reshape(-1),minlength=256).astype(float)
        p=counts[counts>0]/counts.sum()
        result.append(float(-np.sum(p*np.log2(p))))
    return result


def pearson(a,b):
    a=np.asarray(a,dtype=float).reshape(-1);b=np.asarray(b,dtype=float).reshape(-1)
    a=a-a.mean();b=b-b.mean();denom=math.sqrt(float(np.dot(a,a)*np.dot(b,b)))
    return float(np.dot(a,b)/denom) if denom>0 else 0.0


def box7(a):
    padded=np.pad(a,((3,3),(3,3),(0,0)),mode='reflect')
    integral=np.pad(padded,((1,0),(1,0),(0,0))).cumsum(axis=0).cumsum(axis=1)
    return (integral[7:,7:]-integral[:-7,7:]-integral[7:,:-7]+integral[:-7,:-7])/49


def ssim7(a,b):
    if np.array_equal(a,b): return 1.0
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
    mx,my=box7(x),box7(y)
    vx=(box7(x*x)-mx*mx)*49/48;vy=(box7(y*y)-my*my)*49/48
    cov=(box7(x*y)-mx*my)*49/48
    score=((2*mx*my+(0.01*255)**2)*(2*cov+(0.03*255)**2))/((mx*mx+my*my+(0.01*255)**2)*(vx+vy+(0.03*255)**2))
    return float(np.mean(score[3:-3,3:-3]))


def compare(a,b):
    x=a.astype(float);y=b.astype(float);difference=x-y
    mse=float(np.mean(difference*difference));xor=np.bitwise_xor(a,b)
    return {'mse':mse,'psnr_db':float('inf') if mse==0 else 10*math.log10(255**2/mse),
            'ssim_7x7':ssim7(a,b),'pearson_cc':pearson(a,b),
            'npcr_byte_percent':float(100*np.mean(a!=b)),
            'changed_pixel_percent':float(100*np.mean(np.any(a!=b,axis=2))),
            'uaci_percent':float(100*np.mean(np.abs(difference))/255),
            'hamming_bit_percent':float(100*np.mean(np.unpackbits(xor.reshape(-1))))}


def adjacent_correlations(image):
    return {'horizontal':pearson(image[:,:-1],image[:,1:]),
            'vertical':pearson(image[:-1],image[1:]),
            'diagonal':pearson(image[:-1,:-1],image[1:,1:])}


def verify_ssim():
    rng=np.random.default_rng(7373)
    a=rng.integers(0,256,(9,10,3),dtype=np.uint8);b=rng.integers(0,256,(9,10,3),dtype=np.uint8)
    reference=[]
    for row in range(3,6):
        for col in range(3,7):
            for c in range(3):
                x=a[row-3:row+4,col-3:col+4,c].astype(float).reshape(-1)
                y=b[row-3:row+4,col-3:col+4,c].astype(float).reshape(-1)
                mx,my=x.mean(),y.mean();vx,vy=x.var(ddof=1),y.var(ddof=1)
                cov=np.dot(x-mx,y-my)/48
                reference.append(((2*mx*my+6.5025)*(2*cov+58.5225))/((mx*mx+my*my+6.5025)*(vx+vy+58.5225)))
    error=abs(ssim7(a,b)-np.mean(reference))
    assert error<1e-12
    return float(error)
