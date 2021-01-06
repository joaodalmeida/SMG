from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *
from printdata import *

import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
from matplotlib.animation import FuncAnimation
from tensorboardX import SummaryWriter
import matplotlib.animation as animation
import statistics
writer_train = SummaryWriter()
writer_valid = SummaryWriter()
writer_test = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=220,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32, 
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=400, 
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5, 
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=12, 
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='rnn',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_softtt',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='', 
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=7, 
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=100, 
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=50, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=100,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--folder-name', type=str, default='', # load data, like ./paperdata/testset0
                    help='Name of the folder where the data is located.')
parser.add_argument('--patience', type=int, default=5, 
                    help='Number of epochs to wait in validation before early stopping.')
parser.add_argument('--supervised', action='store_true', default=True,
                    help='Supervised models sets relations to Ground Truth.')
parser.add_argument('--burnin_steps', type=int, default=5, 
                    help='Number of time steps in burn in')
parser.add_argument('--easy-train', action='store_true', default=True,
                    help='At train time, give GT position every args.pred_train.') 
parser.add_argument('--pred_train', type=int, default=10,
                    help='Prediction step at train time (if args.easy_train activated)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    args.batch_size, args.suffix, args.folder_name)

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32) #[132,12] 
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32) #[132,12]
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)


if args.encoder == 'mlp':
    encoder = MLPEncoder(args.prediction_steps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
#    mse_train = []
#    mse_1_train = []
#    mse_5_train = []
    mse_10_train = []    
#    mse_25_train = []
#    mse_50_train = []
    mse_1_val = []
    mse_5_val = []
    mse_10_val = []    
    mse_25_val = []
    mse_50_val = []

    encoder.train()
    decoder.train()
    scheduler.step()
    for batch_idx, (data, relations) in enumerate(train_loader): 
        print("TRAINING")
        print(args.batch_size)
        print(batch_idx)
        mse_batch = []
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations) # data is [32,12,50,4], relations are [32,132]

        data_encoder = data[:, :, :args.prediction_steps, :].contiguous()
        if args.decoder == 'rnn':
            data_decoder = data[:, :, -args.prediction_steps-args.burnin_steps+1:-args.prediction_steps+args.pred_train, :].contiguous()
        else:
            data_decoder = data[:, :, -args.prediction_steps:-args.prediction_steps+args.pred_train, :].contiguous()

        optimizer.zero_grad()
        logits = encoder(data_encoder, rel_rec, rel_send) #run encoder that computes proability q of relations z given trajectories x
            #[16,132,2] of values like -1.4073 or 3.4853
        if args.supervised:
            for a in range(len(relations[:,0])):
                for b in range(len(relations[0,:])):
                    if (relations.data[a,b]==1): 
                        logits[a,b,0]=0
                        logits[a,b,1]=10                   
                    else: 
                        logits[a,b,0]=10
                        logits[a,b,1]=0

        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard) #sampling: sample relations z from the concrete reparameterizable approximation of probabilities q
            #[16,132,2]    
        prob = my_softmax(logits, -1) #sampling: sample relations z from the concrete reparameterizable approximation of probabilities q
            #[16,132,2] of values between 0 and 1. From logits' pairs like [-1.1478 -0.0538] or [-1,1499 6.4842] to [0.2509  0.7491] or [0.0004  0.9995] respectively.
            #the greater the difference between the 2 elements of the pair in logits, the greater the difference in prob

        if args.decoder == 'rnn': # run decoder that computes future trajectories
            output = decoder(data_decoder, edges, rel_rec, rel_send, args.pred_train,  #output is [16,12,49,4] for the softt data.
                                burn_in=True,
                                burn_in_steps=args.burnin_steps)
        else:
            output = decoder(data_decoder, edges, rel_rec, rel_send,args.pred_train) 

        if args.decoder == 'rnn':
            output = output[:,:,args.burnin_steps-1:,:]    
            target = data_decoder[:, :, args.burnin_steps:, :] 

        else:
            target = data_decoder[:,:,1:,:]

        loss_nll = nll_gaussian(output[:,:,:,:3], target[:,:,:,:3], args.var) # compute reconstruction error (negative log likelihood)

        if args.prior: # compute KL divergence
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)

        loss = loss_nll + loss_kl # ELBO we want to maximize

        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)
        
        #mse=F.mse_loss(output[:,:,:,:3], target[:,:,:,:3])
#        mse_1=F.mse_loss(output[:,:,:1,:3], target[:,:,:1,:3])
#        mse_5=F.mse_loss(output[:,:,:5,:3], target[:,:,:5,:3])
        mse_10=F.mse_loss(output[:,:,:10,:3], target[:,:,:10,:3])
#        mse_25=F.mse_loss(output[:,:,:25,:3], target[:,:,:25,:3])
#        mse_50=F.mse_loss(output[:,:,:50,:3], target[:,:,:50,:3])
            
        loss.backward() 
        optimizer.step()
        
        for ind in range(data.size(0)):
            mse_batch.append((F.mse_loss(output[ind,:,:10,:3], target[ind,:,:10,:3])).data[0]) #for finding worst and best

        #mse_train.append(mse.data[0])
#        mse_1_train.append(mse_1.data[0])
#        mse_5_train.append(mse_5.data[0])
        mse_10_train.append(mse_10.data[0])
#        mse_25_train.append(mse_25.data[0])
#        mse_50_train.append(mse_50.data[0])
        nll_train.append(loss_nll.data[0])
        kl_train.append(loss_kl.data[0])
        
        if (batch_idx==0) and (epoch==10): 
            plot_guess=(output.data[0,:,:,:3]).cpu().numpy() #[12,49,3]
            plot_target=(target.data[0,:,:,:3]).cpu().numpy() #[12,49,3]

            best_plot_guess=(output.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            best_plot_target=(target.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_guess=(output.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_target=(target.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_guess=(output.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_target=(target.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            random_plot_guess=(output.data[0,:,:,:3]).cpu().numpy()
            random_plot_target=(target.data[0,:,:,:3]).cpu().numpy()
            ## FIGURE: print overlap of trajectories (real and prediction) and relations (real and prediction) for training
            print_frame_3d (plot_target, plot_guess, relations, logits, "Training: trajectories prediction (color) and target (yellow)", args.num_atoms)
            print("accuracy: "+str(acc))
            print_pred_relations(plot_guess,logits,"Training: relations prediction", args.num_atoms)
            ## FIGURE: GIF with movement for the best training case of this batch
            print_gif(best_plot_target,best_plot_guess, "nri_best_model_training.gif", args.num_atoms, args.pred_train) 
            ## FIGURE: GIF with movement for the worst training case of this batch
            print_gif(worst_plot_target,worst_plot_guess, "nri_worst_model_training.gif", args.num_atoms, args.pred_train) 
            #FIGURE: GIF with movement for the median case in this batch
            print_gif(median_plot_target,median_plot_guess, "nri_median_model_training.gif", args.num_atoms, args.pred_train) 
            #FIGURE: GIF with movement for a random case in this batch
            print_gif(random_plot_target,random_plot_guess, "nri_random_model_training.gif", args.num_atoms, args.pred_train)

#        pytorch_total_params = sum(p.numel() for p in decoder.parameters())
#        print("number of parameters: " + str(pytorch_total_params)) # 3979803
#        pytorch_total_trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
#        print("number of trainable parameters: " + str(pytorch_total_trainable_params)) # 3979803
            
    nll_val = []
    acc_val = []
    kl_val = []
    accuracy_matrix=np.zeros((4,args.batch_size))
    accuracy_val = []
    fscore_val = []
#    mse_val = []
    writer_train.add_scalar("Loss/train", loss, epoch)
    writer_train.flush()

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        print("VALIDATION")
        print(args.batch_size)
        print(batch_idx)
        mse_batch=[]
        
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)
    
        data_encoder = data[:, :, :args.prediction_steps, :].contiguous()
        if args.decoder == 'rnn':
            data_decoder = data[:, :, -args.prediction_steps-args.burnin_steps+1:, :].contiguous()
        else:
            data_decoder = data[:,:,-args.prediction_steps:,:].contiguous()

        logits = encoder(data_encoder, rel_rec, rel_send)
        if args.supervised:
            for a in range(len(relations[:,0])): 
                for b in range(len(relations[0,:])):
                    if (relations.data[a,b]==1): 
                        logits[a,b,0]=0
                        logits[a,b,1]=10                     
                    else: 
                        logits[a,b,0]=10
                        logits[a,b,1]=0

        for a in range(len(relations[:,0])):
            for b in range(len(relations[0,:])):
                if ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==11):
                    accuracy_matrix[0,a]+=1 #true positives
                elif ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==0):
                    accuracy_matrix[1,a]+=1 #true negatives
                elif ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==1):
                    accuracy_matrix[2,a]+=1 #false positive
                elif ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==10):
                    accuracy_matrix[3,a]+=1 #false negative
                else: print("ERROR calculating encoder accuracies")
        for a in range(len(relations[:,0])):
            accuracy_val.append((accuracy_matrix[0,a]+accuracy_matrix[1,a])/(accuracy_matrix[0,a]+accuracy_matrix[1,a]+accuracy_matrix[2,a]+accuracy_matrix[3,a]))
            if (accuracy_matrix[0,a]==0):
                fscore_val.append(2/((accuracy_matrix[0,a]+accuracy_matrix[2,a])+(accuracy_matrix[0,a]+accuracy_matrix[3,a])))
            else:
                fscore_val.append(2/(((accuracy_matrix[0,a]+accuracy_matrix[2,a])/accuracy_matrix[0,a])+(accuracy_matrix[0,a]+accuracy_matrix[3,a])/accuracy_matrix[0,a]))
            accuracy_matrix[:,a]=0   
        
        edges = gumbel_softmax(logits, tau=args.temp, hard=True) 
        prob = my_softmax(logits, -1)

        if args.decoder == 'rnn':
            output = decoder(data_decoder, edges, rel_rec, rel_send, 54,  #[16,12,49,4]
                             burn_in=True,
                             burn_in_steps=args.burnin_steps) #
        else:
            output = decoder(data_decoder, edges, rel_rec, rel_send, args.prediction_steps)
         
        if args.decoder == 'rnn':
            output = output [:,:,args.burnin_steps-1:,:] 
            target = data_decoder[:, :, args.burnin_steps:, :]
        else:
            target = data_decoder[:, :, 1:, :]
            
        loss_nll = nll_gaussian(output[:,:,:,:3], target[:,:,:,:3], args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        acc = edge_accuracy(logits, relations)
        acc_val.append(acc)

        #mse = F.mse_loss(output[:,:,:,:3], target[:,:,:,:3])
        mse_1=F.mse_loss(output[:,:,:1,:3], target[:,:,:1,:3])
        mse_5=F.mse_loss(output[:,:,:5,:3], target[:,:,:5,:3])
        mse_10=F.mse_loss(output[:,:,:10,:3], target[:,:,:10,:3])
        mse_25=F.mse_loss(output[:,:,:25,:3], target[:,:,:25,:3])
        mse_50=F.mse_loss(output[:,:,:50,:3], target[:,:,:50,:3])

        for ind in range(data.size(0)):
            mse_batch.append((F.mse_loss(output[ind,:,:50,:3], target[ind,:,:50,:3])).data[0]) #for finding worst and best

        #mse_val.append(mse.data[0])
        mse_1_val.append(mse_1.data[0])
        mse_5_val.append(mse_5.data[0])
        mse_10_val.append(mse_10.data[0])
        mse_25_val.append(mse_25.data[0])
        mse_50_val.append(mse_50.data[0])
        nll_val.append(loss_nll.data[0])
        kl_val.append(loss_kl.data[0])
        
        
        if (batch_idx==0) and (epoch==10):
            plot_guess=(output.data[0,:,:,:3]).cpu().numpy() #[12,49,3]
            plot_target=(target.data[0,:,:,:3]).cpu().numpy() #[12,49,3]
            best_plot_guess=(output.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            best_plot_target=(target.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_guess=(output.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_target=(target.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_guess=(output.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_target=(target.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            random_plot_guess=(output.data[0,:,:,:3]).cpu().numpy()
            random_plot_target=(target.data[0,:,:,:3]).cpu().numpy()
            ## FIGURE: overlap of trajectories (real and prediction) and relations (real and predictions) for validation
            print_frame_3d (plot_target, plot_guess, relations, logits, "Validation: trajectories prediction (color) and target (yellow)", args.num_atoms)
            print("accuracy: "+str(acc))
            ## FIGURE: validation predicted trajectories
            print_pred_trajectories (plot_guess, logits, "Validation: prediction trajectories", args.num_atoms)
            ## FIGURE: Validation target trajectories
            print_real_trajectories (plot_target, relations, "Validation: (real) trajectories target", args.num_atoms)
            ## FIGURE: Validation predicted relations
            print_pred_relations(plot_guess,logits,"Validation: relations prediction", args.num_atoms)
            ## FIGURE: Validation predicted relations
            print_real_relations(plot_target, relations, "Validation: (real) relations target", args.num_atoms)
            #FIGURE: GIF with movement for the best case in this batch
            print_gif(best_plot_target,best_plot_guess, "nri_best_model_validation.gif", args.num_atoms, args.prediction_steps)
            #FIGURE: GIF with movement for the worst case in this batch
            print_gif(worst_plot_target,worst_plot_guess, "nri_worst_model_validation.gif", args.num_atoms, args.prediction_steps)
            #FIGURE: GIF with movement for the median case in this batch
            print_gif(median_plot_target,median_plot_guess, "nri_median_model_validation.gif", args.num_atoms, args.prediction_steps)
            #FIGURE: GIF with movement for a random case in this batch
            print_gif(random_plot_target,random_plot_guess, "nri_random_model_validation.gif", args.num_atoms, args.prediction_steps)

    loss_val = np.mean(nll_val) + np.mean(kl_val) 
    loss_train = np.mean(nll_train) + np.mean(kl_train) 
    writer_train.add_scalars(f'loss/train', {
            'training': loss_train,
            'validation': loss_val,
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
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'mse_10_train: {:.10f}'.format(np.mean(mse_10_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'nll_val: {:.10f}'.format(np.mean(nll_val)),
          'kl_val: {:.10f}'.format(np.mean(kl_val)),
          'mse_1_val: {:.12f}'.format(np.mean(mse_1_val)),
          'mse_5_val: {:.12f}'.format(np.mean(mse_5_val)),
          'mse_10_val: {:.12f}'.format(np.mean(mse_10_val)),
          'mse_25_val: {:.12f}'.format(np.mean(mse_25_val)),
          'mse_50_val: {:.12f}'.format(np.mean(mse_50_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'accuracy_val: {:.10f}'.format(np.mean(accuracy_val)),
          'f1score_val: {:.10f}'.format(np.mean(fscore_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_10_train: {:.10f}'.format(np.mean(mse_10_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_1_val: {:.12f}'.format(np.mean(mse_1_val)),
              'mse_5_val: {:.12f}'.format(np.mean(mse_5_val)),
              'mse_10_val: {:.12f}'.format(np.mean(mse_10_val)),
              'mse_25_val: {:.12f}'.format(np.mean(mse_25_val)),
              'mse_50_val: {:.12f}'.format(np.mean(mse_50_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'accuracy_val: {:.10f}'.format(np.mean(accuracy_val)),
              'f1score_val: {:.10f}'.format(np.mean(fscore_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(nll_val)


def test():
    acc_test = []
    nll_test = []
    kl_test = []
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
    displacement=[[]]
    displacement_batch=[]
    distance=[[]]
    distance_batch=[]
    dis_bef_norm=[]
    tot_mse = 0
    counter = 0
    accuracy_test = []
    fscore_test = []
    accuracy_matrix=np.zeros((4,args.batch_size))
    
    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
    for batch_idx, (data, relations) in enumerate(test_loader):    
        mse_batch=[]
        print("TESTING")
        print (batch_idx)
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        data_encoder = data[:, :, :args.prediction_steps, :].contiguous()
        if args.decoder == 'rnn':
            data_decoder = data[:, :, -args.prediction_steps-args.burnin_steps+1:, :].contiguous()
        else:
            data_decoder = data[:,:,-args.prediction_steps:,:].contiguous()

        logits = encoder(data_encoder, rel_rec, rel_send)
        if args.supervised:
            for a in range(len(relations[:,0])): 
                for b in range(len(relations[0,:])):
                    if (relations.data[a,b]==1): 
                        logits[a,b,0]=0
                        logits[a,b,1]=10                       
                    else: 
                        logits[a,b,0]=10
                        logits[a,b,1]=0

        for a in range(len(relations[:,0])):
            for b in range(len(relations[0,:])):
                if ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==11):
                    accuracy_matrix[0,a]+=1 #true positives
                elif ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==0):
                    accuracy_matrix[1,a]+=1 #true negatives
                elif ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==1):
                    accuracy_matrix[2,a]+=1 #false positive
                elif ((10*relations.data[a,b]+(logits[a,b,1].data[0]>logits[a,b,0].data[0]))==10):
                    accuracy_matrix[3,a]+=1 #false negative
                else: print("ERROR calculating encoder accuracies")
        for a in range(len(relations[:,0])):
            accuracy_test.append((accuracy_matrix[0,a]+accuracy_matrix[1,a])/(accuracy_matrix[0,a]+accuracy_matrix[1,a]+accuracy_matrix[2,a]+accuracy_matrix[3,a]))
            if (accuracy_matrix[0,a]==0):
                fscore_test.append(2/((accuracy_matrix[0,a]+accuracy_matrix[2,a])+(accuracy_matrix[0,a]+accuracy_matrix[3,a])))
            else:
                fscore_test.append(2/(((accuracy_matrix[0,a]+accuracy_matrix[2,a])/accuracy_matrix[0,a])+(accuracy_matrix[0,a]+accuracy_matrix[3,a])/accuracy_matrix[0,a]))
            accuracy_matrix[:,a]=0   

        edges = gumbel_softmax(logits, tau=args.temp, hard=True) 
        prob = my_softmax(logits, -1) 

        if args.decoder == 'rnn':
            output = decoder(data_decoder, edges, rel_rec, rel_send, 54, 
                         burn_in=True,
                         burn_in_steps=args.burnin_steps)
        else:
            output = decoder(data_decoder, edges, rel_rec, rel_send, args.prediction_steps) #[10,12,49,4]
        
        if args.decoder == 'rnn':
            output = output [:,:,args.burnin_steps-1:,:]
            target = data_decoder[:, :, args.burnin_steps:, :] #[10,12,49,4]
        else:
            target = data_decoder[:, :, 1:, :]
        
        loss_nll = nll_gaussian(output[:,:,:,:3], target[:,:,:,:3], args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)
        
        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)
        #mse = F.mse_loss(output[:,:,:,:3], target[:,:,:,:3])
        mse_1=F.mse_loss(output[:,:,:1,:3], target[:,:,:1,:3])
        mse_5=F.mse_loss(output[:,:,:5,:3], target[:,:,:5,:3])
        mse_10=F.mse_loss(output[:,:,:10,:3], target[:,:,:10,:3])
        mse_25=F.mse_loss(output[:,:,:25,:3], target[:,:,:25,:3])
        mse_50=F.mse_loss(output[:,:,:50,:3], target[:,:,:50,:3])

        #mse_test.append(mse.data[0])
        mse_1_test.append(mse_1.data[0])
        mse_5_test.append(mse_5.data[0])
        mse_10_test.append(mse_10.data[0])
        mse_25_test.append(mse_25.data[0])
        mse_50_test.append(mse_50.data[0])
        nll_test.append(loss_nll.data[0])
        kl_test.append(loss_kl.data[0])

        output_plot_50=copy.copy(output)
        target_plot_50=copy.copy(target)
        
        for ind in range(data.size(0)):
            mse_batch.append((F.mse_loss(output[ind,:,:50,:3], target[ind,:,:50,:3])).data[0]) #for finding worst and best

        
        if (batch_idx==0):
            plot_guess=(output.data[0,:,:,:3]).cpu().numpy() #[12,49,3]
            plot_target=(target.data[0,:,:,:3]).cpu().numpy() #[12,49,3]
            best_plot_guess=(output.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            best_plot_target=(target.data[mse_batch.index(min(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            print(max(mse_batch))
            print(mse_batch.index(max(mse_batch)))
            worst_plot_guess=(output.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            worst_plot_target=(target.data[mse_batch.index(max(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_guess=(output.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            median_plot_target=(target.data[mse_batch.index(statistics.median_low(mse_batch)),:,:,:3]).cpu().numpy() #[12,49,3]
            random_plot_guess=(output.data[0,:,:,:3]).cpu().numpy()
            random_plot_target=(target.data[0,:,:,:3]).cpu().numpy()
            ## FIGURE: overlap of trajectories (real and prediction) and relations (real and predictions) for validation
            print_frame_3d (plot_target, plot_guess, relations, logits, "Test: trajectories prediction (color) and target (yellow)", args.num_atoms)
            #FIGURE: GIF with movement for the best case in this batch
            print_gif(best_plot_target,best_plot_guess, "nri_best_model_test.gif", args.num_atoms, args.prediction_steps)
            #FIGURE: GIF with movement for the worst case in this batch
            print_gif(worst_plot_target,worst_plot_guess, "nri_worst_model_test.gif", args.num_atoms, args.prediction_steps)
            #FIGURE: GIF with movement for the median case in this batch
            print_gif(median_plot_target,median_plot_guess, "nri_median_model_test.gif", args.num_atoms, args.prediction_steps)
            #FIGURE: GIF with movement for a random case in this batch
            print_gif(random_plot_target,random_plot_guess, "nri_random_model_test.gif", args.num_atoms, args.prediction_steps)
        
        for ind in range(args.prediction_steps-2):
            ind+=1
            for xx in range(data.size(0)):
                for yy in range(data.size(1)):
                    if ((pow(data_decoder[xx, yy, ind, 0].data[0]-data_decoder[xx, yy, 0, 0].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 1].data[0]-data_decoder[xx, yy, 0, 1].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 2].data[0]-data_decoder[xx, yy, 0, 2].data[0],2))!=0):
                        displacement_=(math.sqrt(pow(data_decoder[xx, yy, ind, 0].data[0]-data_decoder[xx, yy, 0, 0].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 1].data[0]-data_decoder[xx, yy, 0, 1].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 2].data[0]-data_decoder[xx, yy, 0, 2].data[0],2)))
                        mse_=(F.mse_loss(output[xx,yy,ind,:3],target[xx,yy,ind,:3])/displacement_)
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
        
        distance_travelled = np.zeros((data.size(0), data.size(1)))
        for ind in range(args.prediction_steps-2): 
            ind+=1
            for xx in range(data.size(0)): 
                for yy in range(data.size(1)): 
                    if ((pow(data_decoder[xx, yy, ind, 0].data[0]-data_decoder[xx, yy, 0, 0].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 1].data[0]-data_decoder[xx, yy, 0, 1].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 2].data[0]-data_decoder[xx, yy, 0, 2].data[0],2))!=0):
                        distance_travelled[xx,yy]=(distance_travelled[xx,yy]
                              +(math.sqrt(pow(data_decoder[xx, yy, ind, 0].data[0]-data_decoder[xx, yy, ind-1, 0].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 1].data[0]-data_decoder[xx, yy, ind-1, 1].data[0],2)
                              +pow(data_decoder[xx, yy, ind, 2].data[0]-data_decoder[xx, yy, ind-1, 2].data[0],2))))
                        mse_=(F.mse_loss(output[xx,yy,ind,:3],target[xx,yy,ind,:3])/distance_travelled[xx,yy])
                        mse_bef_norm.append(mse_.data[0])
                        dis_bef_norm.append(distance_travelled[xx,yy])
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
        
        for ind in range(args.prediction_steps-1):
            #ind+=1
            for xx in range(data.size(0)): 
                for yy in range(data.size(1)): 
                    mse= F.mse_loss(output[xx,yy,ind,:3],target[xx,yy,ind,:3])
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

    loss_test = np.mean(nll_test)+np.mean(kl_test) 
    writer_test.add_scalar("Loss/test", loss_test, epoch)
    writer_test.flush()

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_1_test: {:.12f}'.format(np.mean(mse_1_test)),
          'mse_5_test: {:.12f}'.format(np.mean(mse_5_test)),
          'mse_10_test: {:.12f}'.format(np.mean(mse_10_test)),
          'mse_25_test: {:.12f}'.format(np.mean(mse_25_test)),
          'mse_50_test: {:.12f}'.format(np.mean(mse_50_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)),
          'accuracy_test: {:.10f}'.format(np.mean(accuracy_test)),
          'f1score_test: {:.10f}'.format(np.mean(fscore_test)))
    #print('MSE: {}'.format(mse_str))
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
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_1_test: {:.12f}'.format(np.mean(mse_1_test)),
              'mse_5_test: {:.12f}'.format(np.mean(mse_5_test)),
              'mse_10_test: {:.12f}'.format(np.mean(mse_10_test)),
              'mse_25_test: {:.12f}'.format(np.mean(mse_25_test)),
              'mse_50_test: {:.12f}'.format(np.mean(mse_50_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              'accuracy_test: {:.10f}'.format(np.mean(accuracy_test)),
              'f1score_test: {:.10f}'.format(np.mean(fscore_test)),
              file=log)
        #print('MSE: {}'.format(mse_str), file=log)
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
