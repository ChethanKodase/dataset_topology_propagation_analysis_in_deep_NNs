import sys
sys.path.append('./')

import math
import numpy as np
from models import AE
import matplotlib.pyplot as plt
from activations import Sin

from layered_models_for_circle import ConvoAE, Autoencoder_linear, VAE_mlp_circle_new, ConvVAE_circle, ConvoAE_for_1024, ConvVAE_circle1024

from loss_functions import contractive_loss_function, loss_fn_mlp_vae, loss_fn_cnn_vae
from torch.autograd import Variable

import ripser
import persim
from persim import plot_diagrams

pi = math.pi

from random import seed
from random import randint
import random
import os

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset


seed(1)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_value = 0
set_seed(seed_value)
#2342


def PointsInCircumNDim(points, transform_to_nD):
    circle_nD = np.matmul(points, transform_to_nD)
    return circle_nD



def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    return distances



dim = 1024
transform_to_nD = 1.2*np.random.rand(3, dim)-2
print(transform_to_nD)


torus3d2kpoints = torch.load('./savedData/3dtorus2000points.pt')
#######################################################################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(torus3d2kpoints[:,0], torus3d2kpoints[:,1], torus3d2kpoints[:,2])
plt.savefig('./results/torusIn3DSpace.png')
plt.close()
#######################################################################


data_tr = torch.from_numpy(PointsInCircumNDim(np.array(torus3d2kpoints), transform_to_nD)).float()
data_val = torch.from_numpy(PointsInCircumNDim(np.array(torus3d2kpoints), transform_to_nD)).float()


A_transform5 = np.random.uniform(-2, 2, 3*1024).reshape(3, 1024)
A_trans_torus = torch.tensor(A_transform5)
data_tr = torch.matmul(torus3d2kpoints, A_trans_torus)

print("A_trans_torus", A_trans_torus)

data_tr = (data_tr - data_tr.mean())/(data_tr.max() - data_tr.mean()) 
data_val = data_tr



hidden_size = 6
no_layers = 2
lr = 5e-3

no_filters = 5
kernel_size = 3
no_layers_conv = 2
latent_dim = 3

model = Autoencoder_linear(latent_dim, dim).to(device)

model_reg_tr = Autoencoder_linear(latent_dim, dim).to(device)
model_reg_ran = Autoencoder_linear(latent_dim, dim).to(device)
model_reg_cheb = Autoencoder_linear(latent_dim, dim).to(device)
model_reg_leg = Autoencoder_linear(latent_dim, dim).to(device)
model_conv = ConvoAE_for_1024(latent_dim).to(device)
model_contra = Autoencoder_linear(latent_dim, dim).to(device)
model_cnn_vae = ConvVAE_circle1024(image_channels=1, h_dim=256, z_dim=latent_dim).to(device)
model_mlp_vae = VAE_mlp_circle_new(image_size=dim, h_dim=6, z_dim=latent_dim).to(device)

no_epochs = 550
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_tr = torch.optim.Adam(model_reg_tr.parameters(), lr=lr)
optimizer_ran = torch.optim.Adam(model_reg_ran.parameters(), lr=lr)
optimizer_cheb = torch.optim.Adam(model_reg_cheb.parameters(), lr=lr)
optimizer_leg = torch.optim.Adam(model_reg_leg.parameters(), lr=lr)

#
optimizer_mlp_vae = torch.optim.Adam(model_mlp_vae.parameters(), lr=1e-3) 

#
optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr =0.002, weight_decay = 1e-5)
optimizer_contra = torch.optim.Adam(model_contra.parameters(), lr =0.002, weight_decay = 1e-5)
optimizer_cnn_vae = torch.optim.Adam(model_cnn_vae.parameters(), lr=1e-3) 

mod_loss = []
mod_loss_tr = []
mod_loss_ran = []
mod_loss_cheb = []
mod_loss_leg = []
mod_loss_conv = []
mod_loss_vae = []
mod_loss_mlp_vae = []


from regularisers_without_vegas import computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes, computeC1Loss_layered

regNodesSamplings = (["mlp_ae", "legendre",
                    "conv", "contra", "mlp_vae", "cnn_vae"])


models = ([model, model_reg_leg,
        model_conv, model_contra, model_mlp_vae, model_cnn_vae])


optimizers = ([optimizer, optimizer_leg, 
            optimizer_conv, optimizer_contra, optimizer_mlp_vae, optimizer_cnn_vae])

szSample = 50
weightJac = False
degPoly=51
alpha = 0.1

batch_size_cfs = 50

no_trn_samples = 50

data_tr = data_tr.float()[:no_trn_samples]
data_tr_ = data_tr.reshape(int(data_tr.shape[0]/batch_size_cfs), batch_size_cfs, dim)


#data_val = data_tr[:2000]
data_val = data_val.float()[:700]

###########################################################
# Legendre
###########################################################
points = np.polynomial.legendre.leggauss(degPoly)[0][::-1]

weights = np.polynomial.legendre.leggauss(degPoly)[1][::-1]
###########################################################
labels = ["mlp_ae", "Reg on legendre nodes","conv", "contra", "mlp_vae", "cnn_vae"]


for ind, model_reg in enumerate(models):
    mod_loss_reg = []
    regNodesSampling = regNodesSamplings[ind]
    print(regNodesSampling)
    # Load the model_reg parameters from ./saved_models

    if(labels[ind]=="mlp_ae" or labels[ind]=="Reg on legendre nodes" or labels[ind]=="contra" ):
        model_reg.load_state_dict(torch.load('./saved_models/model_'+regNodesSampling+'torus_dim_'+str(dim)+'_no_tr_smpls_'+str(no_trn_samples)+'_seed_'+str(seed_value)+'_epochs_'+str(no_epochs)+'_.pt'))
        points_val = (model_reg.encoder_l4(model_reg.encoder_l3(model_reg.encoder_l2(model_reg.encoder_l1(data_val.unsqueeze(1).to(device)))))).detach().cpu().numpy()

        #points_val = points_val.detach().cpu().numpy()
        points_val = points_val.reshape(-1,latent_dim)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points_val[:,0], points_val[:,1], points_val[:,2])
        plt.savefig('./results/'+labels[ind]+'.png')
        plt.close()

        points_val = torch.tensor(points_val)
        dist_matrix = _compute_distance_matrix(points_val, p=2)
        diagrams = ripser.ripser(dist_matrix.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
        plot_diagrams(diagrams, show=True)
        #plot_diagrams(diagrams)
        plt.savefig('./results/'+labels[ind]+'_ph_diag.png')

    if(labels[ind]=="mlp_vae"):
        model_reg.load_state_dict(torch.load('./saved_models/model_'+regNodesSampling+'torus_dim_'+str(dim)+'_no_tr_smpls_'+str(no_trn_samples)+'_seed_'+str(seed_value)+'_epochs_'+str(no_epochs)+'_.pt'))
        points_val = model_reg.fc1(model_reg.encoder_l3(model_reg.encoder_l2(model_reg.encoder_l1(data_val.float().unsqueeze(1).to(device)))))
        points_val = points_val.detach().cpu().numpy()
        points_val = points_val.reshape(-1,latent_dim)

        print('points_val.shape mlpvae', points_val.shape)
        torch.save(points_val, './results/'+labels[ind]+'.pt')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points_val[:,0], points_val[:,1], points_val[:,2])
        plt.savefig('./results/'+labels[ind]+'.png')
        plt.close()

        points_val = torch.tensor(points_val)
        dist_matrix = _compute_distance_matrix(points_val, p=2)
        diagrams = ripser.ripser(dist_matrix.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
        plot_diagrams(diagrams)
        plt.savefig('./results/'+labels[ind]+'_ph_diag.png')

    if(labels[ind]=="cnn_vae" ):
        points_val = model_reg.fc1(model_reg.encoder_l3(model_reg.encoder_l2(model_reg.encoder_l1(data_val.unsqueeze(1).float().to(device)))))
        points_val = points_val.detach().cpu().numpy()

        points_val = points_val.reshape(-1,latent_dim)

        print('points_val.shape cnnvae', points_val.shape)
        torch.save(points_val, './results/'+labels[ind]+'.pt')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points_val[:,0], points_val[:,1], points_val[:,2])
        plt.savefig('./results/'+labels[ind]+'.png')
        plt.close()

        points_val = torch.tensor(points_val)
        dist_matrix = _compute_distance_matrix(points_val, p=2)
        diagrams = ripser.ripser(dist_matrix.cpu().detach().numpy(), distance_matrix=True, maxdim=2)['dgms']
        plot_diagrams(diagrams, show=True)



