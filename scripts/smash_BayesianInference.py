import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import smash
from utilities_public import rescale,sigmoidInv,decompose_covariance
from model_smash import smash_linearMapping
from inferenceFunk import llfunk_iid_Gaussian, llfunk_iLinear_Gaussian, llfunk_AR1Linear_Gaussian,logLikelihood, logPosterior, minusLogPosterior
from MCMC import MCMC_AM,MCMC_options

# Create smash model
setup,mesh=smash.factory.load_dataset('lez')
model = smash.Model(setup, mesh)
# Define bounds for rr_parameters
bounds={'cp': (1e-06, 1000.0), 'ct': (1e-06, 1000.0), 'kexc': (-50, 50), 'llr': (1e-06, 1000.0)}
# Assemble descriptors
d1=1.+0.*model.physio_data.descriptor[:,:,0] # constant, creates an intercept
d2=model.physio_data.descriptor[:,:,1] # drainage density
d3=model.physio_data.descriptor[:,:,5] # soil water storage
descriptors=np.stack([d1,rescale(d2,(-1,1)),rescale(d3,(-1,1))],axis=2) # rescale and stack them into a 3D array
# Define the descriptors used for each rr_parameter (they don't have to be the same for all rr_parameters)
D={'cp':descriptors,'ct':descriptors,'kexc':descriptors,'llr':descriptors}
# Define observed streamflows and uncertainties
qobs=model.response_data.q
qobs[qobs<0]=np.nan # Use nan rather than negative values
qobs_u=qobs*0.1 #model.u_response_data.q_stdev

# Define a smash_linearMapping  model and runs it
myModel=smash_linearMapping(model,D,bounds)
myModel.npar() # Number of parameters 
qsim=myModel.run(theta=1*np.ones(4*3)) # Simulate

# Test inference functions
llfunk_iid_Gaussian(Ysim=qsim,Yobs=qobs,Yu=qobs_u,gamma=np.ones(3))
llfunk_iLinear_Gaussian(Ysim=qsim,Yobs=qobs,Yu=qobs_u,gamma=np.ones(6))
llfunk_AR1Linear_Gaussian(Ysim=qsim,Yobs=qobs,Yu=qobs_u,gamma=np.array([0.5,0.2,0.7,0.5,0.2,0.7,0.5,0.2,0.7]))
parvector=np.array([sigmoidInv(200,bounds['cp']),0,0,
                    sigmoidInv(60,bounds['ct']),0,0,
                    sigmoidInv(2,bounds['kexc']),0,0,
                    sigmoidInv(500,bounds['llr']),0,0,
                    0.5,0.2,0.5,0.2,0.5,0.2])
ll,qsim=logLikelihood(parvector=parvector,Yobs=qobs,Yu=qobs_u,model=myModel)
print(ll)
lpost,lprior,ll,qsim=logPosterior(np.ones(4*3+6),Yobs=qobs,Yu=qobs_u,model=myModel)
print(lpost)

# MCMC strategy
# Start with uniform parameter maps
descriptors=np.stack([d1],axis=2)
D={'cp':descriptors,'ct':descriptors,'kexc':descriptors,'llr':descriptors}
myModel=smash_linearMapping(model,D,bounds)
parvector=np.array([sigmoidInv(200,bounds['cp']),sigmoidInv(60,bounds['ct']),
                    sigmoidInv(2,bounds['kexc']),sigmoidInv(500,bounds['llr']),
                    0.5,0.2,0.5,0.2,0.5,0.2])
foo=minimize(fun=minusLogPosterior,x0=parvector,args=(qobs,qobs_u,myModel),method='Nelder-Mead')
start=foo.x
mcmc=MCMC_AM(logPdf=logPosterior,x0=start,options=MCMC_options(nAdapt=25,nCycles=40),
             Yobs=qobs,Yu=qobs_u,model=myModel)
samples,comps,C,scaleFactor=mcmc
# plt.plot(comps[:,0]);plt.show()
# plt.plot(samples[:,0]);plt.show()
# Get maxpost parameters
maxpost=samples[np.argmax(comps[:,0]),:]
# Get final jump sdevs
sdev,correl=decompose_covariance(C)

# Now include descriptors
descriptors=np.stack([d1,rescale(d2,(-1,1)),rescale(d3,(-1,1))],axis=2) # rescale and stack them into a 3D array
D={'cp':descriptors,'ct':descriptors,'kexc':descriptors,'llr':descriptors}
myModel=smash_linearMapping(model,D,bounds)
# New starting point and starting sdevs
parvector=np.array([maxpost[0],0.,0.,maxpost[1],0.,0.,maxpost[2],0.,0.,maxpost[3],0.,0.,
                    maxpost[4],maxpost[5],maxpost[6],maxpost[7],maxpost[8],maxpost[9]])
jumps=np.array([sdev[0],0.1,0.1,sdev[1],0.1,0.1,sdev[2],0.1,0.1,sdev[3],0.1,0.1,
                sdev[4],sdev[5],sdev[6],sdev[7],sdev[8],sdev[9]])
foo=minimize(fun=minusLogPosterior,x0=parvector,args=(qobs,qobs_u,myModel),method='Nelder-Mead')
start=foo.x
mcmc=MCMC_AM(logPdf=logPosterior,x0=start,C0=np.diag(jumps**2),scaleFactor=scaleFactor,
             options=MCMC_options(nCycles=200),Yobs=qobs,Yu=qobs_u,model=myModel)
samples,comps,C,scaleFactor=mcmc
sdev,correl=decompose_covariance(C)
plt.imshow(correl);plt.show()
ix=slice(int(0.5*len(samples[:,0])),len(samples[:,0]))
plt.plot(comps[ix,0]);plt.show()
plt.plot(samples[ix,0]);plt.show()
plt.plot(samples[ix,1],samples[ix,2]);plt.show()

# AR(1) model
descriptors=np.stack([d1],axis=2)
D={'cp':descriptors,'ct':descriptors,'kexc':descriptors,'llr':descriptors}
myModel=smash_linearMapping(model,D,bounds)
parvector=np.array([sigmoidInv(200,bounds['cp']),sigmoidInv(60,bounds['ct']),
                    sigmoidInv(2,bounds['kexc']),sigmoidInv(500,bounds['llr']),
                    0.5,0.2,0.7,0.5,0.2,0.7,0.5,0.2,0.7])
foo=minimize(fun=minusLogPosterior,x0=parvector,args=(qobs,qobs_u,myModel,llfunk_AR1Linear_Gaussian),method='Nelder-Mead')
start=foo.x
mcmc=MCMC_AM(logPdf=logPosterior,x0=start,options=MCMC_options(nAdapt=50,nCycles=200),
             Yobs=qobs,Yu=qobs_u,model=myModel,llfunk=llfunk_AR1Linear_Gaussian)
samples,comps,C,scaleFactor=mcmc
sdev,correl=decompose_covariance(C)

ix=slice(int(0.5*len(samples[:,0])),len(samples[:,0]))
plt.plot(comps[ix,0]);plt.show()
plt.plot(samples[ix,0]);plt.show()
plt.plot(samples[ix,5],samples[ix,6]);plt.show()

