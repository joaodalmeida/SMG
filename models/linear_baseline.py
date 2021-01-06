from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *
from printdata import *

import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import statistics
from tensorboardX import SummaryWriter
writer_train = SummaryWriter()
writer_valid = SummaryWriter()
writer_test = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True, #False
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, 
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32, 
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--num_atoms', type=int, default=12, 
                    help='Number of atoms in simulation.')
parser.add_argument('--num-layers', type=int, default=2,
                    help='Number of LSTM layers.')
parser.add_argument('--suffix', type=str, default='_softtt', 
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model.')
parser.add_argument('--load-folder', type=str, default='', #load file like ./logs/exp.../
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--dims', type=int, default=7, 
                    help='The number of dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=100,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=50, metavar='N', 
                    help='Num steps to predict before using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--motion', action='store_true', default=False,
                    help='Use motion capture data loader.')
parser.add_argument('--non-markov', action='store_true', default=False,
                    help='Use non-Markovian evaluation setting.') 
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--folder-name', type=str, default='', #where data is located like ./paperdata/valset0
                    help='Name of the folder where the data is located.')
parser.add_argument('--patience', type=int, default=200, #20
                    help='Number of epochs to wait in validation before early stopping.')
parser.add_argument('--baseline', type=str, default='linear',
                    help='Type of baseline model (linear, mlp)')
parser.add_argument('--context', type=int, default=5,
                    help='Temporal context window for each prediction.')
parser.add_argument('--easy-train', action='store_true', default=False,
                    help='At train time, give GT position every args.pred_train.') 
parser.add_argument('--pred_train', type=int, default=10,
                    help='Prediction step at train time (if args.easy_train activated)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    while os.path.isdir(save_folder):
        exp_counter += 1
        save_folder = os.path.join(args.save_folder,
                                   'exp{}'.format(exp_counter))
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'model.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))

else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    args.batch_size, args.suffix, args.folder_name)


class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        # super function inherits from nn.Module so that we can access everything from nn.Module
        super(LinearRegression,self).__init__()
        # Linear function
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x) #, self.linear.weight
    
class MLP_baseline(nn.Module):
    def __init__(self,input_size,output_size,hidden_size):
        # super function inherits from nn.Module so that we can access everything from nn.Module
        super(MLP_baseline,self).__init__()
        self.linear = nn.Linear(input_dim,hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)
    
    def forward(self,x):
        y1 = F.relu(self.linear(x))  
        y2 = F.relu(self.linear2(y1))
        return self.linear3(y2) #, self.linear.weight

input_dim = args.num_atoms*args.context*3+(args.context+1)*4 
output_dim = args.num_atoms*3  # 12 * 3 we want to predict 1 timestep ahead - 3D space trajecotories - of the 12 points
assert(args.timesteps%args.prediction_steps==0)

if (args.baseline=='linear'):
    model = LinearRegression(input_dim, output_dim)
elif (args.baseline=='mlp'):
    model = MLP_baseline(input_dim, output_dim, args.hidden)
else:
    print("MUST DEFINE MODEL TYPE")

if args.load_folder:
    model_file = os.path.join(args.load_folder, 'model.pt')
    model.load_state_dict(torch.load(model_file))
    args.save_folder = False

optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for loss calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)

if args.cuda:
    model.cuda()


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def train(epoch, best_val_loss):
    print("START TRAINING")
    t = time.time()
    loss_train = []
    mse_train = []
    mse_val = []
    mse_1_train = []
    mse_5_train = []
    mse_10_train = []    
    mse_25_train = []
    mse_50_train = []
    mse_1_val = []
    mse_5_val = []
    mse_10_val = []    
    mse_25_val = []
    mse_50_val = []
    in_vector_np = []
    target_vector_np = []
    prediction_step_train=args.prediction_steps

    model.train()
    scheduler.step()
    for batch_idx, (data, relations) in enumerate(train_loader):
        print("training, batch_idx:" + str(batch_idx))
        mse_batch = []
        pred_step_train=args.prediction_steps
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations) #[32,12,100,7]
        optimizer.zero_grad()
     
        
        data_encoder = data[:,:,:args.prediction_steps,:]
        data_decoder =data[:,:,-args.prediction_steps:,:]
    
        context=data_encoder[:,:,-args.context+1:,:3] # [32,12,2,3]
        context=torch.cat([context,data_decoder[:,:,:1,:3]], dim=2) #[32,12,3,3]

        B, N, T, S = data.size(0), data.size(1), data.size(2), data.size(3)
        P = 3
        A = 4
        
        
        for pred_step_unit in range(args.prediction_steps-1):
            context_window = context[:,:,-args.context:] #B, N, C, P    # [32,12,3,3]
            context_window = context_window.contiguous().view(B, N*args.context*P) # B,S #[32,108]
            action = data[:,0,pred_step_unit+args.prediction_steps-args.context:pred_step_unit+args.prediction_steps+1,3:] # B, A # [32,4,4]
            action = action.contiguous().view(B,(args.context+1)*A) # [32,16]
            
            model_input = torch.cat([context_window, action], dim=1) # B, A+S # [32,124]
            
            output = model(model_input) # 32,12*3 # [32,36]
            
            context=torch.cat([context,output.contiguous().view(B,N,1,P)], dim=2)
            
            if (pred_step_unit==0):
                prediction = output.contiguous().view(B,N,1,P)
            else:
                prediction = torch.cat([prediction, output.contiguous().view(B,N,1,P)], dim=2)
        
        target = data_decoder[:,:,1:,:3] # [32,12,49,3]

        loss = nll_gaussian(prediction, target, args.var)


        mse_1=F.mse_loss(prediction[:,:,:1,:], target[:,:,:1,:])
        mse_5=F.mse_loss(prediction[:,:,:5,:], target[:,:,:5,:])
        mse_10=F.mse_loss(prediction[:,:,:10,:], target[:,:,:10,:])
        mse_25=F.mse_loss(prediction[:,:,:25,:], target[:,:,:25,:])
        mse_50=F.mse_loss(prediction[:,:,:50,:], target[:,:,:50,:])
        
        for ind in range(data.size(0)):
            mse_batch.append((F.mse_loss(prediction[ind,:,:10,:], target[ind,:,:10,:])).data[0]) #for finding worst and best
        
        if (batch_idx==0) and (epoch%args.epochs==(args.epochs-1)):
            plot_guess=(prediction.data[0,:,:,:]).cpu().numpy() #[12,49,3]
            plot_target=(target.data[0,:,:,:]).cpu().numpy() #[12,49,3]
            best_plot_guess=(prediction.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            best_plot_target=(target.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_guess=(prediction.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_target=(target.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_guess=(prediction.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_target=(target.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            random_plot_guess=(prediction.data[0,:,:,:3]).cpu().numpy()
            random_plot_target=(target.data[0,:,:,:3]).cpu().numpy()

            ## FIGURE 1: print overlap of trajectories (real and prediction) and relations (real and prediction) for training
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for finger_idx in range(5):
                xs = plot_guess[finger_idx,:,0]
                ys = plot_guess[finger_idx,:,1]
                zs = plot_guess[finger_idx,:,2]
                ax.scatter(xs, ys, zs, s=49)
            for finger_idx in range(5):
                xs = plot_target[finger_idx,:,0]
                ys = plot_target[finger_idx,:,1]
                zs = plot_target[finger_idx,:,2]
                ax.scatter(xs, ys, zs, c="y", s=49)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_title("Training: prediction and target")
            plt.savefig("training_pred_and_target.png")
            plt.show(block=False)
            ## FIGURE: GIF with movement for the best training case of this batch
            print_gif(best_plot_target,best_plot_guess, "nri_best_model_training.gif", args.num_atoms, args.timesteps-pred_step_train)
            ## FIGURE: GIF with movement for the worst training case of this batch
            print_gif(worst_plot_target,worst_plot_guess, "nri_worst_model_training.gif", args.num_atoms, args.timesteps-pred_step_train)
            #FIGURE: GIF with movement for the median case in this batch
            print_gif(median_plot_target,median_plot_guess, "nri_median_model_training.gif", args.num_atoms, args.timesteps-pred_step_train)
            #FIGURE: GIF with movement for a random case in this batch
            print_gif(random_plot_target,random_plot_guess, "nri_random_model_training.gif", args.num_atoms, args.timesteps-pred_step_train)

        loss.backward()
        optimizer.step()
        #mse_train.append(mse.data[0])
        mse_1_train.append(mse_1.data[0])
        mse_5_train.append(mse_5.data[0])
        mse_10_train.append(mse_10.data[0])
        mse_25_train.append(mse_25.data[0])
        mse_50_train.append(mse_50.data[0])
        loss_train.append(loss.data[0])

#        pytorch_total_params = sum(p.numel() for p in model.parameters())
#        print("number of parameters: " + str(pytorch_total_params)) # 3979803
#        pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#        print("number of trainable parameters: " + str(pytorch_total_trainable_params)) # 3979803
#            
    loss_val = []
    mse_val = []
    in_vector_np = []
    target_vector_np = []
    writer_train.add_scalar("Loss/train", loss, epoch)
    writer_train.flush()
    model.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        print("validation, batch_idx:" + str(batch_idx))
        mse_batch = []
        prediction_np =[]
        loss = 0
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, requires_grad=False), Variable(
            relations, requires_grad=False)
        
        data_encoder = data[:,:,:args.prediction_steps,:]
        data_decoder =data[:,:,-args.prediction_steps:,:]
    
        context=data_encoder[:,:,-args.context+1:,:3] # [32,12,2,3]
        context=torch.cat([context,data_decoder[:,:,:1,:3]], dim=2) #[32,12,3,3]

        B, N, T, S = data.size(0), data.size(1), data.size(2), data.size(3)
        P = 3
        A = 4
        
        
        for pred_step_unit in range(args.prediction_steps-1): 
            context_window = context[:,:,-args.context:] #B, N, C, P    # [32,12,3,3]
            context_window = context_window.contiguous().view(B, N*args.context*P) # B,S #[32,108]
            action = data[:,0,pred_step_unit+args.prediction_steps-args.context:pred_step_unit+args.prediction_steps+1,3:] # B, A # [32,4,4]
            action = action.contiguous().view(B,(args.context+1)*A) # [32,16]
            
            model_input = torch.cat([context_window, action], dim=1) # B, A+S # [32,124]
            
            output = model(model_input) # 32,12*3 # [32,36]
            
            context=torch.cat([context,output.contiguous().view(B,N,1,P)], dim=2)
            
            if (pred_step_unit==0):
                prediction = output.contiguous().view(B,N,1,P)
            else:
                prediction = torch.cat([prediction, output.contiguous().view(B,N,1,P)], dim=2)
        
        target = data_decoder[:,:,1:,:3] # [32,12,49,3]

        loss = nll_gaussian(prediction, target, args.var)


        mse_1=F.mse_loss(prediction[:,:,:1,:], target[:,:,:1,:])
        mse_5=F.mse_loss(prediction[:,:,:5,:], target[:,:,:5,:])
        mse_10=F.mse_loss(prediction[:,:,:10,:], target[:,:,:10,:])
        mse_25=F.mse_loss(prediction[:,:,:25,:], target[:,:,:25,:])
        mse_50=F.mse_loss(prediction[:,:,:50,:], target[:,:,:50,:])
        
        for ind in range(data.size(0)):
            mse_batch.append((F.mse_loss(prediction[ind,:,:10,:], target[ind,:,:10,:])).data[0]) #for finding worst and best

        
        loss_val.append(loss.data[0])
        #mse_val.append(mse.data[0])
        mse_1_val.append(mse_1.data[0])
        mse_5_val.append(mse_5.data[0])
        mse_10_val.append(mse_10.data[0])
        mse_25_val.append(mse_25.data[0])
        mse_50_val.append(mse_50.data[0])

        if (batch_idx==0) and (epoch%args.epochs==(args.epochs-1)):
            plot_guess=(prediction.data[0,:,:,:]).cpu().numpy() #[12,49,3]
            plot_target=(target.data[0,:,:,:]).cpu().numpy() #[12,49,3]
            best_plot_guess=(prediction.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            best_plot_target=(target.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_guess=(prediction.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_target=(target.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_guess=(prediction.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_target=(target.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            random_plot_guess=(prediction.data[0,:,:,:]).cpu().numpy()
            random_plot_target=(target.data[0,:,:,:]).cpu().numpy()

            ## FIGURE:
            fig10 = plt.figure()
            ax10 = fig10.add_subplot(111, projection='3d')
            for finger_idx in range(args.num_atoms):
                xs = plot_guess[finger_idx,:,0]
                ys = plot_guess[finger_idx,:,1]
                zs = plot_guess[finger_idx,:,2]
                ax10.scatter(xs, ys, zs, s=49)
            for finger_idx in range(args.num_atoms):
                xs = plot_target[finger_idx,:,0]
                ys = plot_target[finger_idx,:,1]
                zs = plot_target[finger_idx,:,2]
                ax10.scatter(xs, ys, zs, c="y", s=49)
            ax10.set_xlabel('X Label')
            ax10.set_ylabel('Y Label')
            ax10.set_zlabel('Z Label')
            ax10.set_title("Validation: prediction and target")
            plt.savefig("validation_pred_and_target.png")                  
            plt.show(block=False)    
            

            #FIGURE 11: GIF with movement for the best case in this batch
            #print_gif(best_plot_target,best_plot_guess, "nri_best_model_validation.gif", args.num_atoms, args.timesteps-args.prediction_steps)
            #FIGURE 12: GIF with movement for the worst case in this batch
            #print_gif(worst_plot_target,worst_plot_guess, "nri_worst_model_validation.gif", args.num_atoms, args.timesteps-5)
            #FIGURE 13: GIF with movement for the median case in this batch
            #print_gif(median_plot_target,median_plot_guess, "nri_median_model_validation.gif", args.num_atoms, args.timesteps-5)
            #FIGURE 14: GIF with movement for a random case in this batch
            #print_gif(random_plot_target,random_plot_guess, "nri_random_model_validation.gif", args.num_atoms, args.timesteps-5)


    writer_train.add_scalars(f'loss/train', {
            'training': np.mean(loss_train),
            'validation': np.mean(loss_val),
    }, epoch)
    writer_train.flush()
    writer_train.add_scalars(f'mse/train', {
            'training': np.mean(mse_10_train),
            'validation1': np.mean(mse_1_val),
            'validation5': np.mean(mse_5_val),
            'validation10': np.mean(mse_10_val),
            'validation25': np.mean(mse_25_val),
            'validation50': np.mean(mse_50_val),
    }, epoch)
    writer_train.flush()

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(loss_train)),
          'mse_10_train: {:.12f}'.format(np.mean(mse_10_train)),
          #'mse_baseline_train: {:.10f}'.format(np.mean(mse_baseline_train)),
          'nll_val: {:.10f}'.format(np.mean(loss_val)),
          'mse_1_val: {:.12f}'.format(np.mean(mse_1_val)),
          'mse_5_val: {:.12f}'.format(np.mean(mse_5_val)),
          'mse_10_val: {:.12f}'.format(np.mean(mse_10_val)),
          'mse_25_val: {:.12f}'.format(np.mean(mse_25_val)),
          'mse_50_val: {:.12f}'.format(np.mean(mse_50_val)),
          #'mse_baseline_val: {:.10f}'.format(np.mean(mse_baseline_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(loss_val) < best_val_loss:
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(loss_train)),
              'mse_10_train: {:.12f}'.format(np.mean(mse_10_train)),
              #'mse_baseline_train: {:.10f}'.format(np.mean(mse_baseline_train)),
              'nll_val: {:.10f}'.format(np.mean(loss_val)),
              'mse_1_val: {:.12f}'.format(np.mean(mse_1_val)),
              'mse_5_val: {:.12f}'.format(np.mean(mse_5_val)),
              'mse_10_val: {:.12f}'.format(np.mean(mse_10_val)),
              'mse_25_val: {:.12f}'.format(np.mean(mse_25_val)),
              'mse_50_val: {:.12f}'.format(np.mean(mse_50_val)),
              #'mse_baseline_val: {:.10f}'.format(np.mean(mse_baseline_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(loss_val)


def test():
    loss_test = []
    mse_baseline_test = []
    mse_test = []
    mse_1_test = []
    mse_5_test = []
    mse_10_test = []    
    mse_25_test = []
    mse_50_test = []
    mse_bef_norm=[]
    cumulative_mse_bef_norm = []
    mse_normalized_to_distance_batch=[]
    mse_normalized_to_length_batch=[]
    mse_normalized_to_displacement_batch=[]
    mse_cumulative_batch=[0]
    mse_normalized_to_distance=[[]]
    mse_normalized_to_length=[[]]
    mse_normalized_to_displacement=[[]]
    cumulative_mse=[[]]
    in_vector_np = []
    target_vector_np = []
    displacement=[[]]
    displacement_batch=[]
    distance=[[]]
    distance_batch=[]
    dis_bef_norm=[]
    tot_mse = 0
    tot_mse_baseline = 0
    counter = 0

    model.eval()
    model.load_state_dict(torch.load(model_file))
    for batch_idx, (inputs, relations) in enumerate(test_loader):
        print("TESTING")
        print(batch_idx)
        mse_batch=[]
        prediction_np =[]
        loss = 0

        #assert (inputs.size(2) - args.timesteps) >= args.timesteps

        if args.cuda:
            inputs = inputs.cuda()
        else:
            inputs = inputs.contiguous()
        inputs = Variable(inputs, volatile=True) #[32,12,100,7]
        


        data_encoder = inputs[:,:,:args.prediction_steps,:]
        data_decoder =inputs[:,:,-args.prediction_steps:,:]
    
        context=data_encoder[:,:,-args.context+1:,:3] # [32,12,2,3]
        context=torch.cat([context,data_decoder[:,:,:1,:3]], dim=2) #[32,12,3,3]

        B, N, T, S = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
        P = 3
        A = 4
        
        
        for pred_step_unit in range(args.prediction_steps-1): 
            context_window = context[:,:,-args.context:] #B, N, C, P    # [32,12,3,3]
            context_window = context_window.contiguous().view(B, N*args.context*P) # B,S #[32,108]
            action = inputs[:,0,pred_step_unit+args.prediction_steps-args.context:pred_step_unit+args.prediction_steps+1,3:] # B, A # [32,4,4]
            action = action.contiguous().view(B,(args.context+1)*A) # [32,16]
            
            model_input = torch.cat([context_window, action], dim=1) # B, A+S # [32,124]
            
            output = model(model_input) # 32,12*3 # [32,36]
            
            context=torch.cat([context,output.contiguous().view(B,N,1,P)], dim=2)
            
            if (pred_step_unit==0):
                prediction = output.contiguous().view(B,N,1,P)
            else:
                prediction = torch.cat([prediction, output.contiguous().view(B,N,1,P)], dim=2)
        
        target = data_decoder[:,:,1:,:3] # [32,12,49,3]

        loss = nll_gaussian(prediction, target, args.var)


        #mse = F.mse_loss(output[:,:,:,:], target[:,:,:,:])
        mse_1=F.mse_loss(prediction[:,:,:1,:], target[:,:,:1,:])
        mse_5=F.mse_loss(prediction[:,:,:5,:], target[:,:,:5,:])
        mse_10=F.mse_loss(prediction[:,:,:10,:], target[:,:,:10,:])
        mse_25=F.mse_loss(prediction[:,:,:25,:], target[:,:,:25,:])
        mse_50=F.mse_loss(prediction[:,:,:50,:], target[:,:,:50,:])
        
        for ind in range(inputs.size(0)):
            mse_batch.append((F.mse_loss(prediction[ind,:,:10,:], target[ind,:,:10,:])).data[0]) #for finding worst and best

        
        loss_test.append(loss.data[0])
        #mse_test.append(mse.data[0])
        mse_1_test.append(mse_1.data[0])
        mse_5_test.append(mse_5.data[0])
        mse_10_test.append(mse_10.data[0])
        mse_25_test.append(mse_25.data[0])
        mse_50_test.append(mse_50.data[0])

        if (batch_idx==0):
            plot_guess=(prediction.data[0,:,:,:]).cpu().numpy() #[12,49,3]
            plot_target=(target.data[0,:,:,:]).cpu().numpy() #[12,49,3]
            best_plot_guess=(prediction.data[mse_batch.index(min(mse_batch)),:,:,:]).cpu().numpy() #[12,49,3]
            best_plot_target=(target.data[mse_batch.index(min(mse_batch)),:,:,:]).cpu().numpy() #[12,49,3]
            worst_plot_guess=(prediction.data[mse_batch.index(max(mse_batch)),:,:,:]).cpu().numpy() #[12,49,3]
            worst_plot_target=(target.data[mse_batch.index(max(mse_batch)),:,:,:]).cpu().numpy() #[12,49,3]
            median_plot_guess=(prediction.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:]).cpu().numpy() #[12,49,3]
            median_plot_target=(target.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:]).cpu().numpy() #[12,49,3]
            random_plot_guess=(prediction.data[0,:,:,:]).cpu().numpy()
            random_plot_target=(target.data[0,:,:,:]).cpu().numpy()
            #FIGURE: GIF with movement for the best case in this batch
            print_gif(best_plot_target,best_plot_guess, "nri_best_model_test.gif", args.num_atoms, args.timesteps-args.prediction_steps)
            #FIGURE: GIF with movement for the worst case in this batch
            print_gif(worst_plot_target,worst_plot_guess, "nri_worst_model_test.gif", args.num_atoms, args.timesteps-args.prediction_steps)
            #FIGURE: GIF with movement for the median case in this batch
            print_gif(median_plot_target,median_plot_guess, "nri_median_model_test.gif", args.num_atoms, args.timesteps-args.prediction_steps)
            #FIGURE: GIF with movement for a random case in this batch
            print_gif(random_plot_target,random_plot_guess, "nri_random_model_test.gif", args.num_atoms, args.timesteps-args.prediction_steps)
        
        for ind in range(args.prediction_steps-1):
            #ind+=1
            for xx in range(inputs.size(0)):
                for yy in range(inputs.size(1)):
                    if ((pow(target[xx, yy, ind, 0].data[0]-target[xx, yy, 0, 0].data[0],2)
                              +pow(target[xx, yy, ind, 1].data[0]-target[xx, yy, 0, 1].data[0],2)
                              +pow(target[xx, yy, ind, 2].data[0]-target[xx, yy, 0, 2].data[0],2))!=0):
                        displacement_=(math.sqrt(pow(target[xx, yy, ind, 0].data[0]-target[xx, yy, 0, 0].data[0],2)
                              +pow(target[xx, yy, ind, 1].data[0]-target[xx, yy, 0, 1].data[0],2)
                              +pow(target[xx, yy, ind, 2].data[0]-target[xx, yy, 0, 2].data[0],2)))
                        mse_=(F.mse_loss(prediction[xx,yy,ind,:],target[xx,yy,ind,:])/displacement_)
                        mse_bef_norm.append(mse_.data[0])
                        dis_bef_norm.append(displacement_)
            mse_normalized_to_displacement_batch.append(np.mean(mse_bef_norm))
            displacement_batch.append(np.mean(dis_bef_norm))
            mse_bef_norm=[]
            dis_bef_norm=[]
        if batch_idx == 0:
            mse_normalized_to_displacement[0]=mse_normalized_to_displacement_batch
            displacement[0]=displacement_batch
        else:
            mse_normalized_to_displacement.append(mse_normalized_to_displacement_batch)
            displacement.append(displacement_batch)
        mse_normalized_to_displacement_batch=[]
        displacement_batch=[]
        
        distance_travelled = np.zeros((inputs.size(0), inputs.size(1)))
        for ind in range(args.prediction_steps-1):
            #ind+=1
            for xx in range(inputs.size(0)): #32
                for yy in range(inputs.size(1)): #12
                    if ((pow(target[xx, yy, ind, 0].data[0]-target[xx, yy, 0, 0].data[0],2)
                              +pow(target[xx, yy, ind, 1].data[0]-target[xx, yy, 0, 1].data[0],2)
                              +pow(target[xx, yy, ind, 2].data[0]-target[xx, yy, 0, 2].data[0],2))!=0):
                        distance_travelled[xx,yy]=(distance_travelled[xx,yy]
                              +(math.sqrt(pow(target[xx, yy, ind, 0].data[0]-target[xx, yy, ind-1, 0].data[0],2)
                              +pow(target[xx, yy, ind, 1].data[0]-target[xx, yy, ind-1, 1].data[0],2)
                              +pow(target[xx, yy, ind, 2].data[0]-target[xx, yy, ind-1, 2].data[0],2))))
                        mse_=(F.mse_loss(prediction[xx,yy,ind,:],target[xx,yy,ind,:])/distance_travelled[xx,yy])
                        dis_bef_norm.append(distance_travelled[xx,yy])
                        mse_bef_norm.append(mse_.data[0])
            mse_normalized_to_distance_batch.append(np.mean(mse_bef_norm))
            distance_batch.append(np.mean(dis_bef_norm))
            mse_bef_norm=[]
            dis_bef_norm=[]
        if batch_idx == 0:
            mse_normalized_to_distance[0]=mse_normalized_to_distance_batch
            distance[0]=distance_batch
        else:
            mse_normalized_to_distance.insert(batch_idx, mse_normalized_to_distance_batch)
            distance.insert(batch_idx, distance_batch)
        mse_normalized_to_distance_batch=[]
        distance_batch=[]
        
        for ind in range(args.prediction_steps-1): #48
            #ind+=1
            for xx in range(inputs.size(0)): #32
                for yy in range(inputs.size(1)): #12
                    mse= F.mse_loss(prediction[xx,yy,ind,:3],target[xx,yy,ind,:3])
                    cumulative_mse_bef_norm.append(mse.data[0])
                    mse_=mse/0.7185 #0.7185 será o comprimento de um dedo
                    mse_bef_norm.append(mse_.data[0])
            mse_cumulative_batch.append(np.mean(cumulative_mse_bef_norm)+mse_cumulative_batch[-1])
            cumulative_mse_bef_norm=[]
            mse_normalized_to_length_batch.append(np.mean(mse_bef_norm))
            mse_bef_norm=[]
        if batch_idx == 0:
            cumulative_mse[0]=mse_cumulative_batch
            mse_normalized_to_length[0]=mse_normalized_to_length_batch
        else:
            mse_normalized_to_length.insert(batch_idx,mse_normalized_to_length_batch)
            cumulative_mse.insert(batch_idx,mse_cumulative_batch)
        mse_normalized_to_length_batch=[]
        mse_cumulative_batch=[0]


    mse_normalized_to_displacement_batch = [float(sum(col))/len(col) for col in zip(*mse_normalized_to_displacement)]
    mse_normalized_to_distance_batch = [float(sum(col))/len(col) for col in zip(*mse_normalized_to_distance)]
    mse_normalized_to_length_batch = [float(sum(col))/len(col) for col in zip(*mse_normalized_to_length)]
    displacement_batch = [float(sum(col))/len(col) for col in zip(*displacement)]
    distance_batch = [float(sum(col))/len(col) for col in zip(*distance)]
    cumulative_mse_batch = [float(sum(col))/len(col) for col in zip(*cumulative_mse)]

    writer_test.add_scalar("Loss/test", np.mean(loss_test), epoch)
    writer_test.flush()

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(loss_test)),
          'mse_test: {:.12f}'.format(np.mean(mse_test)),
          'mse_1_test: {:.12f}'.format(np.mean(mse_1_test)),
          'mse_5_test: {:.12f}'.format(np.mean(mse_5_test)),
          'mse_10_test: {:.12f}'.format(np.mean(mse_10_test)),
          'mse_25_test: {:.12f}'.format(np.mean(mse_25_test)),
          'mse_50_test: {:.12f}'.format(np.mean(mse_50_test)),
          'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)))
#    print('MSE: {}'.format(mse_str))
    #print('MSE Baseline: {}'.format(mse_baseline_str))
    print('MSE normalized to displacement: {}'.format(mse_normalized_to_displacement_batch))
    print('MSE normalized to distance: {}'.format(mse_normalized_to_distance_batch))
    print('MSE normalized to length: {}'.format(mse_normalized_to_length_batch))
    print('travelled distance: {}'.format(distance_batch))
    print('displacement: {}'.format(displacement_batch))
    print('cumulative error: {}'.format(cumulative_mse_batch))
    
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(loss_test)),
              'mse_test: {:.12f}'.format(np.mean(mse_test)),
              'mse_1_test: {:.12f}'.format(np.mean(mse_1_test)),
              'mse_5_test: {:.12f}'.format(np.mean(mse_5_test)),
              'mse_10_test: {:.12f}'.format(np.mean(mse_10_test)),
              'mse_25_test: {:.12f}'.format(np.mean(mse_25_test)),
              'mse_50_test: {:.12f}'.format(np.mean(mse_50_test)),
              'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)),
              file=log)
#        print('MSE: {}'.format(mse_str), file=log)
        #print('MSE Baseline: {}'.format(mse_baseline_str), file=log)
        print('MSE normalized to displacement: {}'.format(mse_normalized_to_displacement_batch), file=log)
        print('MSE normalized to distance: {}'.format(mse_normalized_to_distance_batch), file=log)
        print('MSE normalized to length: {}'.format(mse_normalized_to_length_batch), file=log)
        print('travelled distance: {}'.format(distance_batch), file=log)
        print('displacement: {}'.format(displacement_batch), file=log)
        print('cumulative error: {}'.format(cumulative_mse_batch), file=log)
        log.flush()


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
increasing_loss=0
old_val_loss=0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss>old_val_loss:
        increasing_loss+=1
    else:
        increasing_loss=0
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
    old_val_loss=val_loss
    if (epoch>best_epoch+args.patience):
        print("Early stopping: patience exceeded")
        break
    if (increasing_loss>=args.patience/2):
        print("Early stopping: loss is going up")
        break
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
test()
writer_train.close()
writer_valid.close()
writer_test.close()
if log is not None:
    print(save_folder)
    log.close()

plt.show()
