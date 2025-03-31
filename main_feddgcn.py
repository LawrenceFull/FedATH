import argparse
import warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from util.task_util import accuracy
from util.base_util import seed_everything, load_dataset
from model import GCN, edge_mask
from tqdm import tqdm
import torch.nn.functional as F
import os
import logging
import datetime
from tqdm import tqdm
import torch_geometric

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# experimental environment setup
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--algorithm', type=str, default='fedath')
parser.add_argument('--root', type=str, default='./dataset')  #E:/FedTAD-main/dataset
parser.add_argument('--log_root', type=str, default='./LOG')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="Cora")
parser.add_argument('--partition', type=str, default="Louvain", choices=["Louvain", "Metis"])
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--num_dims', type=int, default=64)
parser.add_argument('--loss_divergence', type=float, default=7)
parser.add_argument('--loss_recons', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=512)

args = parser.parse_args()

def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))  / emb1.shape[0]
    return HSIC

def negative_entropy(predicted_y):
    predicted_y = F.softmax(predicted_y)
    return torch.sum(predicted_y*torch.log(predicted_y))

if __name__ == "__main__":
    if os.path.exists(args.log_root):
        pass
    else:
        os.makedirs(args.log_root) 

    log_name = args.algorithm + '_experiments_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.log_root, log_name),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info('{} running, parti_num:{}, dataset:{}, lr:{}, num_epochs:{}, loss_divergence:{}'.format(
                                                                                    args.algorithm,
                                                                                    args.num_clients,
                                                                                    args.dataset,
                                                                                    args.lr,
                                                                                    args.num_epochs,
                                                                                    args.loss_divergence
                                                                                    ))

    
    seed_everything(seed=args.seed)
    dataset = load_dataset(args)
    loss_ce = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{args.gpu_id}")

    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]

    local_models_c = [ GCN(feat_dim=subgraphs[client_id].x.shape[1], 
                         hid_dim=args.hid_dim, 
                         out_dim=dataset.num_classes,
                         dropout=args.dropout).to(device)
                    for client_id in range(args.num_clients) ]

    local_models_b = [ GCN(feat_dim=subgraphs[client_id].x.shape[1],
                        hid_dim=args.hid_dim,
                        out_dim=dataset.num_classes,
                        dropout=args.dropout).to(device)
                    for client_id in range(args.num_clients) ]

    local_masks = [ edge_mask(input_dim=subgraphs[client_id].x.shape[1],
                    hidden_dim=args.hid_dim,
                    out_dim=dataset.num_classes,
                    dropout_rate=args.dropout).to(device)
                  for client_id in range(args.num_clients) ]


    local_optimizers_c = [Adam([{'params':local_models_c[client_id].parameters(), 'lr':args.lr},
                                {'params':local_masks[client_id].parameters(), 'lr':args.lr}], 
                            weight_decay=args.weight_decay) for client_id in range(args.num_clients)]
  
    local_optimizers_b = [Adam(local_models_b[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) for client_id in range(args.num_clients)]
    
    global_model = GCN(feat_dim=subgraphs[0].x.shape[1], 
                         hid_dim=args.hid_dim, 
                         out_dim=dataset.num_classes,
                         dropout=args.dropout).to(device)
    
    best_server_val = 0
    best_server_test = 0
    best_server_test_b = 0

    
    for round_id in tqdm(range(args.num_rounds)):

        # global model broadcast
        for client_id in range(args.num_clients):
            local_models_c[client_id].load_state_dict(global_model.state_dict())
        
        # local train
        for client_id in range(args.num_clients):
            for epoch_id in range(args.num_epochs):
                local_models_c[client_id].train()
                local_models_b[client_id].train()
                local_masks[client_id].train()

                edge_scores_c = local_masks[client_id](subgraphs[client_id])
                edge_scores_b = (1 - edge_scores_c).detach()
               
                logits_c = local_models_c[client_id].forward(subgraphs[client_id], edge_scores=edge_scores_c)
               
                logits_b = local_models_b[client_id].forward(subgraphs[client_id], edge_scores=edge_scores_b)

                loss_divergence_c = loss_dependence(logits_c, logits_b.detach(), subgraphs[client_id].x.shape[0])
                loss_divergence_b = loss_dependence(logits_c.detach(), logits_b, subgraphs[client_id].x.shape[0])

                loss_ce_c = loss_ce(logits_c[subgraphs[client_id].train_idx], 
                               subgraphs[client_id].y[subgraphs[client_id].train_idx])
                loss_c = loss_ce_c + args.loss_divergence * loss_divergence_c

                loss_nce_b = negative_entropy(logits_b)
                loss_b = loss_nce_b + args.loss_divergence * loss_divergence_b

                local_optimizers_c[client_id].zero_grad()
                loss_c.backward()
                local_optimizers_c[client_id].step()

                local_optimizers_b[client_id].zero_grad()
                loss_b.backward()
                local_optimizers_b[client_id].step()
                
        # global aggregation 
        with torch.no_grad():
            for client_id in range(args.num_clients):
                weight = subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] 
                for (local_state, global_state) in zip(local_models_c[client_id].parameters(), global_model.parameters()):
                    if client_id == 0:
                        global_state.data = weight * local_state
                    else:
                        global_state.data += weight * local_state
                
        # global eval
        global_acc_val = 0
        global_acc_test = 0
        global_acc_test_b = 0
        global_f1_test = 0
        with torch.no_grad():
            for client_id in range(args.num_clients):
                edge_scores_c = local_masks[client_id](subgraphs[client_id])

                logits = local_models_c[client_id].forward(subgraphs[client_id], edge_scores=edge_scores_c)

                loss_train = loss_ce(logits[subgraphs[client_id].train_idx], 
                                subgraphs[client_id].y[subgraphs[client_id].train_idx])

                loss_test = loss_ce(logits[subgraphs[client_id].test_idx], 
                                subgraphs[client_id].y[subgraphs[client_id].test_idx])
                acc_train, f1_train = accuracy(logits[subgraphs[client_id].train_idx], 
                                subgraphs[client_id].y[subgraphs[client_id].train_idx])

                acc_test, f1_test = accuracy(logits[subgraphs[client_id].test_idx], 
                                subgraphs[client_id].y[subgraphs[client_id].test_idx])

                
                logger.info(f"[client {client_id}]: acc_train: {acc_train:.2f}\tacc_test: {acc_test:.2f}\tloss_train: {loss_train:.4f}\tloss_test: {loss_test:.4f}")
                global_acc_test += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_test
                global_f1_test += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * f1_test

        logger.info(f"[server]: current_round: {round_id}\tglobal_acc_val: {global_acc_val:.2f}\tglobal_acc_test: {global_acc_test:.2f}\tglobal_f1_test: {global_f1_test:.2f}")

        logger.info("*"*50)

        

        
        

        
        
      