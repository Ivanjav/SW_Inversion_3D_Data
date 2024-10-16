#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:01:57 2021

@author: ivan
"""
import numpy as np
import pandas as pd
import math as mt
import xarray as xr
import scipy as sp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def dispersion(c,vs,vp,rho,h,f):
    n=np.size(vs)
    K=np.zeros((2*n,2*n))
    w=2*mt.pi*f
    k=w/c
    r=np.sqrt(1-(c**2)/(vp**2),dtype=complex)
    s=np.sqrt(1-(c**2)/(vs**2),dtype=complex)
    D=2*(1-np.cosh(k*r[0:-1]*h)*np.cosh(k*s[0:-1]*h))\
      +(1/(r[0:-1]*s[0:-1])+(r[0:-1]*s[0:-1]))*\
      np.sinh(k*r[0:-1]*h)*np.sinh(k*s[0:-1]*h)
    D=np.real(D)
    for j in range(0,n-1):
        k11=np.real(((k*rho[j]*c**2)/D[j])*((np.power(s[j],-1))*\
             np.cosh(k*r[j]*h[j])*np.sinh(k*s[j]*h[j])-r[j]*\
             np.sinh(k*r[j]*h[j])*np.cosh(k*s[j]*h[j])))
        k12=np.real(((k*rho[j]*c**2)/D[j])*(np.cosh(k*r[j]*h[j])*\
            np.cosh(k*s[j]*h[j])-r[j]*s[j]*np.sinh(k*r[j]*h[j])\
            *np.sinh(k*s[j]*h[j])-1)-k*rho[j]*(vs[j]**2)*(1+s[j]**2))
        k13=np.real(((k*rho[j]*c**2)/D[j])*(r[j]*np.sinh(k*r[j]*h[j])-\
            (s[j]**(-1))*np.sinh(k*s[j]*h[j])))
        k14=np.real(((k*rho[j]*c**2)/D[j])*(-np.cosh(k*r[j]*h[j])+\
            np.cosh(k*s[j]*h[j])))
        k21=k12
        k22=np.real(((k*rho[j]*c**2)/D[j])*((r[j]**(-1))*\
             np.sinh(k*r[j]*h[j])*np.cosh(k*s[j]*h[j])-s[j]*\
             np.cosh(k*r[j]*h[j])*np.sinh(k*s[j]*h[j])))
        k23=-k14
        k24=np.real(((k*rho[j]*c**2)/D[j])*(-(r[j]**(-1))*\
             np.sinh(k*r[j]*h[j])+s[j]*np.sinh(k*s[j]*h[j])))
        k31=k13
        k32=k23
        k33=k11
        k34=-k12
        k41=k14
        k42=k24
        k43=-k21
        k44=k22
        K11=np.array([[k11,k12],[k21,k22]])
        K12=np.array([[k13,k14],[k23,k24]])  
        K21=np.array([[k31,k32],[k41,k42]])
        K22=np.array([[k33,k34],[k43,k44]])
        
        if j==0:
           rows = np.arange(0,2)
           columns = np.arange(0,4)
           K[rows[:, np.newaxis], columns]=np.concatenate((K11,K12),axis=1).reshape((2,4))
           K21a=K21
           K22a=K22
        else :
           rows=np.arange(2,4)+2*(j-1)
           columns=np.arange(0,6)+2*(j-1)
           K[rows[:, np.newaxis], columns]=np.concatenate((K21a,K22a+K11,K12),axis=1).reshape((2,6))
           K21a=K21
           K22a=K22
           
    k11e=(k*rho[n-1]*(vs[n-1]**2))*((r[n-1]*(1-s[n-1]**2))/(1-r[n-1]*s[n-1]))
    k12e=(k*rho[n-1]*(vs[n-1]**2))*((1-s[n-1]**2)/(1-r[n-1]*s[n-1]))-\
         2*k*rho[n-1]*(vs[n-1]**2)
    k21e=k12e
    k22e=(k*rho[n-1]*(vs[n-1]**2))*((s[n-1]*(1-s[n-1]**2))/(1-r[n-1]*s[n-1]))
    
    Ke=np.array([[k11e,k12e],[k21e,k22e]])
    rows = np.arange(2*n-2,2*n)
    columns = np.arange(2*n-4,2*n)
    K=K.astype('complex')
    K[rows[:, np.newaxis], columns]=np.concatenate((K21,K22+Ke),axis=1).reshape((2,4))
    
    return(np.real(np.linalg.det(K)))       
       
def forward_dispersion(vs,vp,rho,h,f):
    cR=f*0
    c0=0.88*vs[0]    
    for i in range(0,np.size(f)):    
        cR[i] = np.abs(fsolve(dispersion, x0=c0,  args=(vs,vp,rho,h,f[i]), xtol=1e-10))
       # c0 = 0.5*cR[i]
    return(cR)
        
def eigenfuctions(vs,vp,rho,h,dh,f,c):   
    H=2*(c/f)
    hx=h
    h=dh*np.ones(np.round(H/dh).astype('int'))
    Nx=np.append(np.round(hx/dh).astype('int'),[1])
    if np.size(h)>=np.sum(Nx):
        N=Nx
        N[-1]=np.size(h)-np.sum(Nx)+1
    else :
        N=Nx[np.cumsum(Nx)<np.size(h)]
        N=np.append(N,np.size(h)-np.sum(N))
    w=2*mt.pi*f
    k=w/c
    G=np.eye(4)
    
    
    P=np.zeros((4,4))
    for n in range(0,np.size(hx)):
        gamma=np.sqrt((k**2-(w/vp[n])**2).astype('complex'))
        v=np.sqrt((k**2-(w/vs[n])**2).astype('complex'))
        mu=(vs[n]**2)*rho[n]
        P[0,0]=np.real(1+(2*mu/((w**2)*rho[n]))*(2*(k**2)*(np.sinh(gamma*hx[n]/2))**2-(k**2+v**2)*(np.sinh(v*hx[n]/2))**2))
        P[2,2]=P[0,0]
        P[0,1]=np.real((k*mu/((w**2)*rho[n]))*((k**2+v**2)*np.sinh(gamma*hx[n])/gamma-2*v*np.sinh(v*hx[n])))
        P[3,2]=-P[0,1]
        P[0,2]=np.real((1/((w**2)*rho[n]))*((k**2)*np.sinh(gamma*hx[n])/gamma-v*np.sinh(v*hx[n])))
        P[0,3]=np.real((2*k/((w**2)*rho[n]))*((np.sinh(gamma*hx[n]/2))**2-(np.sinh(v*hx[n]/2))**2))
        P[1,2]=-P[0,3]
        P[1,0]=np.real((k*mu/((w**2)*rho[n]))*((k**2+v**2)*np.sinh(v*hx[n])/v-2*gamma*np.sinh(gamma*hx[n])))
        P[2,3]=-P[1,0]
        P[1,1]=np.real(1+(2*mu/((w**2)*rho[n]))*(2*(k**2)*(np.sinh(v*hx[n]/2))**2-(k**2+v**2)*(np.sinh(gamma*hx[n]/2))**2))
        P[3,3]=P[1,1]
        P[1,3]=np.real((1/((w**2)*rho[n]))*((k**2)*np.sinh(v*hx[n])/v-gamma*np.sinh(gamma*hx[n])))
        P[2,0]=np.real(((mu**2)/((w**2)*rho[n]))*(4*(k**2)*gamma*np.sinh(gamma*hx[n])-((k**2+v**2)**2)*np.sinh(v*hx[n])/v))
        P[2,1]=np.real(2*(mu**2)*(k**2+v**2)*P[0,3])
        P[3,0]=-P[2,1]
        P[3,1]=np.real(((mu**2)/((w**2)*rho[n]))*(4*(k**2)*v*np.sinh(v*hx[n])-((k**2+v**2)**2)*np.sinh(gamma*hx[n])/gamma))
        G=np.matmul(P,G)
        
        
    P=G
    gamma=np.sqrt(k**2-(w/vp[-1])**2)
    v=np.sqrt(k**2-(w/vs[-1])**2)
    mu=(vs[-1]**2)*rho[-1]
    #M1=np.array([[mt.exp(gamma*np.sum(hx)),0,0,0],
    #             [0,mt.exp(v*np.sum(hx)),0,0],
    #             [0,0,mt.exp(-gamma*np.sum(hx)),0],
    #             [0,0,0,mt.exp(-v*np.sum(hx))]])
    M2=np.array([[2*vs[-1]*mu*k*gamma*v,-vs[-1]*mu*v*(k**2+v**2),-vs[-1]*k*v,vs[-1]*gamma*v],
                 [-vp[-1]*mu*gamma*(k**2+v**2),2*vp[-1]*mu*k*gamma*v,vp[-1]*gamma*v,-vp[-1]*k*gamma],
                 [2*vs[-1]*mu*k*gamma*v,vs[-1]*mu*v*(k**2+v**2),vs[-1]*k*v,vs[-1]*gamma*v],
                 [-vp[-1]*mu*gamma*(k**2+v**2),-2*vp[-1]*mu*k*gamma*v,-vp[-1]*gamma*v,-vp[-1]*k*gamma]]).reshape((4,4))
    F=M2
    B=np.matmul(F,P)
    r=np.zeros((4,np.sum(N)+1))
    r[1,0]=1
    r[2,0]=0
    r[3,0]=0
    r[0,0]=-(B[2,1]/B[2,0])
        
    r0=np.array([-(B[2,1]/B[2,0]),1,0,0]).T
    iz=0
    for n in range(0,np.size(N)):  
        gamma=np.sqrt((k**2-(w/vp[n])**2).astype('complex'))
        v=np.sqrt((k**2-(w/vs[n])**2).astype('complex'))  
        mu=(vs[n]**2)*rho[n]
        for i in range (0,N[n]):
            P[0,0]=np.real(1+(2*mu/((w**2)*rho[n]))*(2*(k**2)*(np.sinh(gamma*np.sum(h[np.arange(0,i+1)])/2))**2-(k**2+v**2)*(np.sinh(v*np.sum(h[np.arange(0,i+1)])/2))**2))
            P[2,2]=P[0,0]
            P[0,1]=np.real((k*mu/((w**2)*rho[n]))*((k**2+v**2)*np.sinh(gamma*np.sum(h[np.arange(0,i+1)]))/gamma-2*v*np.sinh(v*np.sum(h[np.arange(0,i+1)]))))
            P[3,2]=-P[0,1]
            P[0,2]=np.real((1/((w**2)*rho[n]))*((k**2)*np.sinh(gamma*np.sum(h[np.arange(0,i+1)]))/gamma-v*np.sinh(v*np.sum(h[np.arange(0,i+1)]))))
            P[0,3]=np.real((2*k/((w**2)*rho[n]))*((np.sinh(gamma*np.sum(h[np.arange(0,i+1)])/2))**2-(np.sinh(v*np.sum(h[np.arange(0,i+1)])/2))**2))
            P[1,2]=-P[0,3]
            P[1,0]=np.real((k*mu/((w**2)*rho[n]))*((k**2+v**2)*np.sinh(v*np.sum(h[np.arange(0,i+1)]))/v-2*gamma*np.sinh(gamma*np.sum(h[np.arange(0,i+1)]))))
            P[2,3]=-P[1,0]
            P[1,1]=np.real(1+(2*mu/((w**2)*rho[n]))*(2*(k**2)*(np.sinh(v*np.sum(h[np.arange(0,i+1)])/2))**2-(k**2+v**2)*(np.sinh(gamma*np.sum(h[np.arange(0,i+1)])/2))**2))
            P[3,3]=P[1,1]
            P[1,3]=np.real((1/((w**2)*rho[n]))*((k**2)*np.sinh(v*np.sum(h[np.arange(0,i+1)]))/v-gamma*np.sinh(gamma*np.sum(h[np.arange(0,i+1)]))))
            P[2,0]=np.real(((mu**2)/((w**2)*rho[n]))*(4*(k**2)*gamma*np.sinh(gamma*np.sum(h[np.arange(0,i+1)]))-((k**2+v**2)**2)*np.sinh(v*np.sum(h[np.arange(0,i+1)]))/v))
            P[2,1]=np.real(2*(mu**2)*(k**2+v**2)*P[0,3])
            P[3,0]=-P[2,1]
            P[3,1]=np.real(((mu**2)/((w**2)*rho[n]))*(4*(k**2)*v*np.sinh(v*np.sum(h[np.arange(0,i+1)]))-((k**2+v**2)**2)*np.sinh(gamma*np.sum(h[np.arange(0,i+1)]))/gamma))
            r[:,iz+1]=np.matmul(P,r0)
            iz=iz+1
        r0=r[:,iz]
    
    r1=r[0,:]
    r2=r[1,:]
    r3=r[2,:]
    r4=r[3,:]
    z=np.concatenate((np.array([0]),np.cumsum(h)),axis=0)
    return(r1,r2,r3,r4,z,N)
    
def analytic_jacobian(vs,vp,rho,h,dh,f,c):
    mux=(vs**2)*rho
    lambx=(vp**2-2*(vs**2))*rho
    J=np.zeros((np.size(f),np.size(vs)))
    for j in range (0,np.size(f)):
        w=2*mt.pi*f[j]
        k=w/c[j]
        (r1,r2,r3,r4,z,N)=eigenfuctions(vs,vp,rho,h,dh,f[j],c[j])
        N[-1]=N[-1]+1
        rhox=np.zeros(np.sum(N))
        mu=np.zeros(np.sum(N))
        lamb=np.zeros(np.sum(N))
        iz=0
        for i in range(0,np.size(N)):
            mu[np.arange(iz,np.sum(N[np.arange(0,i+1)]))]=mux[i]*np.ones(N[i])
            lamb[np.arange(iz,np.sum(N[np.arange(0,i+1)]))]=lambx[i]*np.ones(N[i])
            rhox[np.arange(iz,np.sum(N[np.arange(0,i+1)]))]=rho[i]*np.ones(N[i])
            iz=np.sum(N[np.arange(0,i+1)])
        
    
        dr1=k*r2+r3/mu
        dr2=-(k*lamb/(lamb+2*mu))*r1 + (1/(lamb+2*mu))*r4
        f1=(1/2)*(rhox*(r1**2+r2**2))
        f2=(1/2)*((lamb+2*mu)*(r1**2)+mu*(r2**2))
        f3=(lamb*(r1*dr2) - mu*(r2*dr1))
        I1=np.trapz(f1,z)
        I2=np.trapz(f2,z)
        I3=np.trapz(f3,z)
        UR=(I2+I3/(2*k))/(c[j]*I1)
    
        fJ=((k*r2-dr1)**2-4*k*r1*dr2)
        N=np.concatenate((np.array([0]),N))
    
    
        for i in range (0,np.size(N)-1):
            rangeN1=np.arange(0,i+1)
            rangeN2=np.arange(0,i+2)
            rangez=np.arange(np.sum(N[rangeN1]),np.sum(N[rangeN2]))
            J[j,i]=(rho[i]*vs[i]/(2*(k**2)*UR*I1))*np.trapz(fJ[rangez],z[rangez])
    return(J)

def sw_inversion(c,f,vsi,rho,h,rps,n_iter,alpha,dh):
    import sys
    if any('jupyter' in arg for arg in sys.argv):
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
    vpi=rps*vsi
    c_modeled=forward_dispersion(vsi,vpi,rho,h,f)
    e=np.zeros(n_iter+1)
    e[0]=np.linalg.norm(c-c_modeled)
    for k in tqdm(range(n_iter)):       
        nf=np.size(f)
        Js=analytic_jacobian(vsi,vpi,rho,h,dh,f,c_modeled)
        L=np.diag((nf/np.sqrt(np.std(c-c_modeled, ddof=1)))*np.ones(nf))
        A=L.dot(Js)
        U,S,V = np.linalg.svd(A,full_matrices=False)
        d=L.dot((c-c_modeled).T)
        #M=V*np.linalg.inv(np.diag(S).dot(np.diag(S).T)+alpha*np.eye(np.size(S)))
        M=np.linalg.solve(np.diag(S).dot(np.diag(S).T)+alpha*np.eye(np.size(S)),V).T
        dvs=np.linalg.multi_dot([M, np.diag(S), U.T, d])
        vsi=vsi+dvs.T
        vpi=rps*vsi       
        c_modeled=forward_dispersion(vsi,vpi,rho,h,f)
        e[k+1]=np.linalg.norm(c-c_modeled)
        if (e[k+1] < 1e-5) :
            break       
        if (k>5) and (np.mean(np.abs(np.diff(e[k-4:k+1])))) < 1 :
            break
    return(vsi,vpi,e)

def adjLRT(da,p):
    f=np.array(da[da.dims[1]])
    x=np.array(da[da.dims[0]])
    x=x.reshape(np.size(x),1)
    p=p.reshape(1,np.size(p))
    d=da.data
    m=np.zeros((np.size(p),np.size(f)),dtype=complex)
    L=np.zeros((np.size(x),np.size(p)),dtype=complex)
    for i in range(0,np.size(f)):
        L=np.exp(1j*2*mt.pi*f[i]*x.dot(p))         
        m[:,i]=L.T.dot(d[:,i])
    return(m)

def DispersionSpectrum(data,v,fmin,fmax,Nf,pstep):
    from scipy.interpolate import interp1d
    time=np.array(data[data.dims[1]])
    dt=np.diff(time[1:3])
    FXdata= np.fft.fft(data.data,Nf,axis=1)
    freq = np.fft.fftfreq(Nf, d=dt)
    dx=xr.DataArray(FXdata, coords=[np.array(data[data.dims[0]]), freq], dims=["offset", "freq"])
    p=np.arange(1/v[-1],1/v[0]+pstep,pstep)
    da=dx.where((dx.freq >= fmin )*(dx.freq <= fmax), drop=True)
    mp=adjLRT(da,p)
    
    vp=1/p
    m=np.zeros((np.size(v),np.size(da.freq)), dtype=complex)
    for i in range(0,np.size(da.freq)):
        #fun = sp.interpolate.interp1d(vp,mp[:,i])
        #m[:,i] = fun(v)
        #funR = sp.interpolate.interp1d(vp,np.real(mp[:,i]),fill_value="extrapolate")
        #funI = sp.interpolate.interp1d(vp,np.imag(mp[:,i]),fill_value="extrapolate")
        funR = interp1d(vp,np.real(mp[:,i]),fill_value="extrapolate")
        funI = interp1d(vp,np.imag(mp[:,i]),fill_value="extrapolate")
        m[:,i] = funR(v)+1j*funI(v)
       
    m=np.nan_to_num(m)
    D=xr.DataArray(m, coords=[v, da.freq], dims=["vel", "freq"])
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(np.abs(D.data)**2)
    D.data = scaler.transform(np.abs(D.data)**2)
    return(D)
   
def ManualPicking(D,cmap):
    plt.figure()
    D.plot.imshow(D.dims[1],D.dims[0],cmap=cmap, origin='lower')
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    pts=np.asarray(plt.ginput(n=-1,timeout=-1)) 
    plt.close()
    return(pts)

def InitialModel(pts,Nh):
    f=np.linspace(pts[0,0],pts[-1,0],Nh)
    fun=sp.interpolate.interp1d(pts[:,0],pts[:,1],fill_value="extrapolate")
    L=np.round(0.63*(fun(f)/f))
    hi = np.diff(np.flip(L))
    zi=np.append(0,np.cumsum(hi))
    vsi=np.flip(fun(f))/0.88
    zi = np.append(zi,zi[-1]+L[0]-L[1])
    return(vsi,hi,zi)

def AutomaticPicking(D,threshold,eps,num,fig,reg,color='jet'):
    # Clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.cluster import DBSCAN
    ind = np.array(np.where(D.data>threshold)).T
    X1 = np.array(D[D.dims[1]][ind[:,1]])
    X2 = np.array(D[D.dims[0]][ind[:,0]])
    rows=ind[:,0].astype('int')
    columns=ind[:,1].astype('int')
    A=D.data
    B=A[rows,columns]
    Xdf = pd.DataFrame(np.concatenate(([X1],[X2],[B]), axis=0).T, columns=['freq','vel','amp'])
    Xdf=Xdf.sort_values(by=['freq'])
    X = Xdf[['freq','vel']].to_numpy()
    scaler = StandardScaler()
    poly = PolynomialFeatures(5)
    scaler.fit(poly.fit_transform(X))
    X_scaled = scaler.transform(poly.fit_transform(X))
    dbscan = DBSCAN(eps=eps)
    clusters = dbscan.fit_predict(X_scaled)
    # Classification
    df1 = pd.DataFrame(np.concatenate((clusters.reshape(np.size(clusters),1),X_scaled),axis=1))
    df2 = df1.groupby([0]).mean().sum(axis=1).sort_values()
    idx=np.array(np.where(clusters==df2.index[0]))
    idx=idx.reshape(np.size(idx,1))
    A=Xdf['amp'].to_numpy()
    Xobs=pd.DataFrame(np.concatenate((X[idx,:],A[idx].reshape(np.size(idx),1)),axis=1), columns=['freq','vel','amp'])
    ind = Xobs.groupby(['freq'])['amp'].transform(max) == Xobs['amp']
    Xobs=Xobs[ind]
    Xobs=Xobs.drop(columns=['amp'])
    # Regression
    if reg:
        from sklearn.linear_model import LinearRegression
        poly = PolynomialFeatures(5)
        scaler.fit(poly.fit_transform(Xobs['freq'].to_numpy().reshape(-1,1)))
        X_scaled = scaler.transform(poly.fit_transform(Xobs['freq'].to_numpy().reshape(-1,1)))
        model = LinearRegression().fit(X_scaled, Xobs['vel'])
        Yobs = model.predict(X_scaled)
        idy=np.asarray(np.where(np.sign(np.diff(Yobs))==-1)).ravel()
        Xobs=Xobs.to_numpy()
        fsta=Xobs[idy[0],0]
        fend=Xobs[idy[-1],0]
        freq=np.linspace(fsta, fend, num, endpoint=True)
        #Xobs=Xobs.to_numpy()
        #Xobs=Xobs[idy,:]
        #Xobs[:,1]=Yobs[idy]
        pts=np.zeros((num,2))
        cR=model.predict(scaler.transform(poly.fit_transform(freq.reshape(-1,1))))
        pts[:,0]=freq
        pts[:,1]=cR
    else :
        Xobs=Xobs.to_numpy()
        #freq=np.linspace(Xobs[0,0], Xobs[-1,0], num, endpoint=True)
        freq=D.freq.values[D.freq.values<Xobs[-1,0]]
        pts=np.zeros((len(freq),len(freq)))
        pts[:,0]= freq
        for i in range(len(freq)):
            pts[i,1]=D.vel[np.argmax(D.data[:,i])]
        #pts=Xobs.to_numpy()
    if fig :
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_title("Binarization (threshold=0.8)")
        axs[0, 0].plot(X[:, 0], X[:, 1],'o',color='red')
        axs[0, 1].set_title(f"Clustering with DBSCAN (eps={eps})")
        axs[0, 1].scatter(X[:, 0], X[:, 1], c=clusters, s=60)
        axs[1, 0].set_title("Cluster selection")
        D.plot.imshow(D.dims[1],D.dims[0],cmap=color, origin='lower',add_colorbar=False, ax=axs[1,0])
        axs[1, 0].plot(X[idx,0],X[idx,1],'o',color='k')
        axs[1, 1].set_title("Points selection")
        D.plot.imshow(D.dims[1],D.dims[0],cmap=color, origin='lower',add_colorbar=False, ax=axs[1,1])
        axs[1, 1].plot(pts[:,0],pts[:,1],'o--',color='k')
        for ax in axs.flat:
            ax.set(xlabel='Frequency (Hz)', ylabel='Phase velocity (m/s)')
    return(pts)


def adj_beam_LRT(data, dt, x, y, p, theta, fmin, fmax):
    """
    Compute the beamformed data using an optimized approach.

    Parameters:
    data : ndarray
        Input data array (time x receivers).
    dt : float
        Time sampling interval.
    x : array-like
        x-coordinates of receivers.
    y : array-like
        y-coordinates of receivers.
    p : ndarray
        Slowness values.
    theta : ndarray
        Angle values in degrees.
    fmin : float
        Minimum frequency.
    fmax : float
        Maximum frequency.

    Returns:
    m : ndarray
        Beamformed data.
    f : ndarray
        Frequencies used.
    Pw : ndarray
        Power spectrum.
    """
    from scipy.fft import fft, fftfreq
    # Convert x and y to NumPy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Step 1: Convert theta from degrees to radians
    theta = theta * 2 * np.pi / 360

    # Step 2: Sampling frequency
    #fs = 1 / dt

    # Step 3: Compute FFT of the data (along axis 0, padding to 8001)
    Data = fft(data, 10001,axis=0)

    # Step 4: Get the dimensions of the transformed data
    Nf, Nr = Data.shape

    # Step 5: Frequency vector
    #freq = np.arange(0, Nf) / Nf * (fs/2)
    freq = fftfreq(Nf,dt)

    # Step 7: Lengths of p and theta
    M = len(p)
    N = len(theta)

    # Step 8: Compute Lx (precompute for all j and l)
    Lx = np.zeros((len(x), N * M), dtype=complex)

    for j in range(M):
        for l in range(N):
            Lx[:, l + j * N] =  p[j] * (x*np.sin(theta[l]) + y*np.cos(theta[l]))
    Lx = Lx.T
    # Step 9: Find indices where frequency is between fmin and fmax
    ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    f = freq[ind]

    # Step 10: Compute matrix `m`
    m = np.zeros((N*M, len(ind)), dtype=complex)

    # Step 6: Transpose Data
    d = Data[ind,:].T  # This transposes the matrix

    for k in range(len(f)):
        w0 = 2 * np.pi * f[k]
        #L = np.power(Lx, w0)
        L = np.exp(-1j*w0*Lx)
        m[:, k] = L @ d[:, k]

    # Step 11: Reshape `m` into 3D array and compute the power spectrum
    P = m.reshape(M, N, len(f))
    Pw = np.abs(P) ** 2

    return m, f, Pw

def beamDispersion(data, dt, x, y, p, theta, fmin, fmax, minpeak=0.9):
    """
    Compute the beamformed data using an optimized approach.

    Parameters:
    data : ndarray
        Input data array (time x receivers).
    dt : float
        Time sampling interval.
    x : array-like
        x-coordinates of receivers.
    y : array-like
        y-coordinates of receivers.
    p : ndarray
        Slowness values.
    theta : ndarray
        Angle values in degrees.
    fmin : float
        Minimum frequency.
    fmax : float
        Maximum frequency.

    Returns:
    m : ndarray
        Beamformed data.
    f : ndarray
        Frequencies used.
    Pw : ndarray
        Power spectrum.
    """
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks
    from scipy.interpolate import interp2d

    # Convert x and y to NumPy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Step 1: Convert theta from degrees to radians
    theta = theta * 2 * np.pi / 360

    # Step 2: Sampling frequency
    #fs = 1 / dt

    # Step 3: Compute FFT of the data (along axis 0, padding to 8001)
    Data = fft(data, 10001,axis=0)

    # Step 4: Get the dimensions of the transformed data
    Nf, Nr = Data.shape

    # Step 5: Frequency vector
    #freq = np.arange(0, Nf) / Nf * (fs/2)
    freq = fftfreq(Nf,dt)

    # Step 7: Lengths of p and theta
    M = len(p)
    N = len(theta)

    # Step 8: Compute Lx (precompute for all j and l)
    Lx = np.zeros((len(x), N * M), dtype=complex)

    for j in range(M):
        for l in range(N):
            Lx[:, l + j * N] =  p[j] * (x*np.sin(theta[l]) + y*np.cos(theta[l]))
    Lx = Lx.T
    # Step 9: Find indices where frequency is between fmin and fmax
    ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    f = freq[ind]

    # Step 10: Compute matrix `m`
    m = np.zeros((N*M, len(ind)), dtype=complex)

    # Step 6: Transpose Data
    d = Data[ind,:].T  # This transposes the matrix

    for k in range(len(f)):
        w0 = 2 * np.pi * f[k]
        #L = np.power(Lx, w0)
        L = np.exp(-1j*w0*Lx)
        m[:, k] = L @ d[:, k]

    # Step 11: Reshape `m` into 3D array and compute the power spectrum
    P = m.reshape(M, N, len(f))
    Pw = np.abs(P) ** 2
    sumP = np.max(Pw, axis=1)  # Summing along the first axis
    Pn = (sumP - np.min(sumP, axis=0)) / (np.max(sumP, axis=0) - np.min(sumP, axis=0))


    # Initialize empty lists for xp and xf
    xp = np.array([])
    xf = np.array([])

    # Assume Pw_LRT, f_LRT, and p are already defined as numpy arrays
    for idf in range(len(f)):
        # Take the maximum along the first axis (axis=0)
        Pw_LRTn = np.max(Pw[:, :, idf], axis=1)
        
        # Normalize Pw_LRTn
        Pw_LRTn = Pw_LRTn / np.max(Pw_LRTn)
        
        # Find local maxima where Pw_LRTn > 0.9
        peaks, _ = find_peaks(Pw_LRTn)
        idp = peaks[Pw_LRTn[peaks] > minpeak]
        
        # Append corresponding values to xp and xf
        xp = np.append(xp, p[idp])  # Append values from p[idp] to xp
        xf = np.append(xf, np.full(len(idp), f[idf]))  # Repeat freq[idf] and append to xf
        Xobs=np.column_stack((xf, 1/xp))

    return Pn, f, Xobs

def decreasing(vel, freq):
    """
    Removes elements from 1D NumPy arrays vel and freq where vel does not follow a strictly decreasing pattern.
    
    Parameters:
    vel (numpy.ndarray): Input 1D NumPy array representing velocities.
    freq (numpy.ndarray): Input 1D NumPy array representing frequencies, with the same length as vel.

    Returns:
    numpy.ndarray, numpy.ndarray: New 1D arrays of vel and freq with elements removed where vel does not follow a decreasing pattern.
    """
    if len(vel) == 0 or len(freq) == 0 or len(vel) != len(freq):
        raise ValueError("Input arrays must be non-empty and have the same length.")
    
    # Initialize the result lists with the first elements
    decreasing_vel = [vel[0]]
    corresponding_freq = [freq[0]]
    
    # Iterate through the vel array and keep only elements that are smaller than the previous one
    for i in range(1, len(vel)):
        if vel[i] <= decreasing_vel[-1]:
            decreasing_vel.append(vel[i])
            corresponding_freq.append(freq[i])  # Keep the corresponding freq value
    # Convert the result lists to numpy arrays before stacking
    decreasing_vel = np.array(decreasing_vel)
    corresponding_freq = np.array(corresponding_freq)
    
    # Convert the results back to numpy arrays
    return np.hstack((corresponding_freq[:, np.newaxis], decreasing_vel[:, np.newaxis])) 


# Define simplified RGB values for parula colormap
from matplotlib.colors import LinearSegmentedColormap
parula_data = [
    (0.2081, 0.1663, 0.5292),
    (0.2116, 0.1898, 0.5777),
    (0.2123, 0.2138, 0.6270),
    (0.2081, 0.2386, 0.6771),
    (0.1959, 0.2645, 0.7279),
    (0.1707, 0.2919, 0.7792),
    (0.1253, 0.3242, 0.8303),
    (0.0591, 0.3598, 0.8683),
    (0.0117, 0.3875, 0.8820),
    (0.0060, 0.4086, 0.8828),
    (0.0165, 0.4266, 0.8786),
    (0.0329, 0.4430, 0.8720),
    (0.0498, 0.4586, 0.8641),
    (0.0629, 0.4737, 0.8554),
    (0.0723, 0.4887, 0.8467),
    (0.0779, 0.5040, 0.8384),
    (0.0793, 0.5200, 0.8312),
    (0.0749, 0.5375, 0.8263),
    (0.0641, 0.5570, 0.8240),
    (0.0488, 0.5772, 0.8228),
    (0.0343, 0.5966, 0.8199),
    (0.0265, 0.6137, 0.8135),
    (0.0239, 0.6287, 0.8038),
    (0.0231, 0.6418, 0.7913),
    (0.0228, 0.6535, 0.7768),
    (0.0267, 0.6642, 0.7607),
    (0.0384, 0.6743, 0.7436),
    (0.0590, 0.6838, 0.7254),
    (0.0843, 0.6928, 0.7062),
    (0.1133, 0.7015, 0.6859),
    (0.1453, 0.7098, 0.6646),
    (0.1801, 0.7177, 0.6424),
    (0.2178, 0.7250, 0.6193),
    (0.2586, 0.7317, 0.5954),
    (0.3022, 0.7376, 0.5712),
    (0.3482, 0.7424, 0.5473),
    (0.3953, 0.7459, 0.5244),
    (0.4420, 0.7481, 0.5033),
    (0.4871, 0.7491, 0.4840),
    (0.5300, 0.7491, 0.4661),
    (0.5709, 0.7485, 0.4494),
    (0.6099, 0.7473, 0.4337),
    (0.6473, 0.7456, 0.4188),
    (0.6834, 0.7435, 0.4044),
    (0.7184, 0.7411, 0.3905),
    (0.7525, 0.7384, 0.3768),
    (0.7858, 0.7356, 0.3633),
    (0.8185, 0.7327, 0.3498),
    (0.8507, 0.7299, 0.3360),
    (0.8824, 0.7274, 0.3217),
    (0.9139, 0.7258, 0.3063),
    (0.9450, 0.7261, 0.2886),
    (0.9739, 0.7314, 0.2666),
    (0.9938, 0.7455, 0.2403),
    (0.9990, 0.7653, 0.2164),
    (0.9955, 0.7861, 0.1967),
    (0.9880, 0.8066, 0.1794),
    (0.9789, 0.8271, 0.1633),
    (0.9697, 0.8481, 0.1475),
    (0.9626, 0.8705, 0.1309),
    (0.9589, 0.8949, 0.1132),
    (0.9598, 0.9218, 0.0948),
    (0.9661, 0.9514, 0.0755),
    (0.9763, 0.9831, 0.0538),
]

# Create custom colormap
parula_cmap = LinearSegmentedColormap.from_list('parula', parula_data)
