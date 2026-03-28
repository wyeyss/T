import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
from copy import deepcopy
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
from meta_patch import *
from reconstruction import *
from pathlib import Path
from model.TSFormer.TSmodel import *
from utils import *
import datetime


class PatchFSL(nn.Module):
    """
    Full PatchFSL Model
    """
    def __init__(self, data_args, model_args, task_args, PatchFSL_cfg, model='GWN'):
        # PatchFSL_cfg :{data_list, s im, K, patch_encoder, base_dir}
        super(PatchFSL,self).__init__()

        self.data_args, self.model_args, self.task_args, self.PatchFSL_cfg = data_args, model_args, task_args, PatchFSL_cfg
        # Pattern_Encoder
        
        pattern = torch.load(PatchFSL_cfg['base_dir'] / 'pattern/{}/{}_{}_cl.pt'.format(PatchFSL_cfg["data_list"],PatchFSL_cfg["sim"],PatchFSL_cfg["K"]))
        pattern = pattern.detach().to(self.PatchFSL_cfg['device'])
        
        Pattern_Encoder = PatternEncoder_patternkeyv2(pattern)

        # FC_model
        model_count={'patch':2, 'pattern':2}
        num_model = model_count[PatchFSL_cfg['patch_encoder']]
        FCmodel = FCNet(num_model * model_args['mae']['out_channel'], model_args['mae']['out_channel'], task_args['maml']['pred_num'])

        # ST_model
        if(model == "GWN"):
            STmodel = BatchA_patch_gwnet(out_dim = model_args['mae']['out_channel'],supports_len = 2)
        else:
            raise NotImplementedError
        # Patch_model
        Reconsmodel = ReconstrucAdjNet(model_args['mae']['out_channel'])

        EncoderLayer = nn.TransformerEncoderLayer(d_model = 128, nhead=2, dim_feedforward = 128)
        Pattern_Day = nn.TransformerEncoder(EncoderLayer, num_layers = 1)
        
        # model_list
        self.model_list = nn.ModuleList([Pattern_Day, Pattern_Encoder, STmodel, FCmodel, Reconsmodel])

        print("[INFO]Pattern_Day has {} params, STmodel has {} params, FCmodel has {} params, Reconsmodel has {} params, Pattern_Encoder has {} params".format(count_parameters(Pattern_Day),count_parameters(STmodel), count_parameters(FCmodel),count_parameters(Reconsmodel), count_parameters(Pattern_Encoder)))
    
    def forward(self, data_i, A, stage = 'train'):
        Pattern_Day, Pattern_Encoder, STmodel, FCmodel, Reconsmodel = self.model_list
        
        if(stage == 'train'):
            Pattern_Encoder.train()
            Pattern_Day.train()
            FCmodel.train()
            STmodel.train()
            Reconsmodel.train()
        else:
            Pattern_Day.eval()
            Pattern_Encoder.eval()
            FCmodel.eval()
            STmodel.eval()
            Reconsmodel.eval()
        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        # print("x shape is : {}, y shape is : {}".format(x.shape, y.shape))
        x, y, means, stds, A = torch.tensor(x).to(self.PatchFSL_cfg['device']), torch.tensor(y).to(self.PatchFSL_cfg['device']),torch.tensor(means).to(self.PatchFSL_cfg['device']),torch.tensor(stds).to(self.PatchFSL_cfg['device']),torch.tensor(A,dtype=torch.float32).to(self.PatchFSL_cfg['device'])
        # remember that the input of TSFormer is [B, N, 2, L]
        x = x.permute(0,1,3,2)
        # shape : [B, N, 12, 1]
        raw_x = x[:,:,0:1, -12:].permute(0,1,3,2)
        raw_x_1day = x[:,:,0:1, -288:].permute(0,1,3,2)
        
        B, N, ___, __ = raw_x_1day.shape
        raw_x_1day = raw_x_1day.reshape(B, N, 24, 12)
        # raw_feature shape : [B, N, D]
        
        if(self.PatchFSL_cfg['patch_encoder'] == 'pattern'):
            # H : [B, N, 24, 12]
            H = raw_x_1day
            # day1pattern : [B, N, 24, D]
            day1pattern = Pattern_Encoder(H)
            BB, NN, LL, DD = day1pattern.shape
            # day1pattern : [L, BN, D]
            day1pattern = day1pattern.reshape(BB * NN, LL, DD).permute(1,0,2)
            # pattern : [L, BN, D]
            pattern = Pattern_Day(day1pattern, mask=None)
            # pattern : [B, N, D]
            pattern = pattern[-1:, :, :].squeeze(0).reshape(BB, NN, DD)
            
            
            A_patch = Reconsmodel(pattern)
        
        A_list = [A_patch, A_patch.permute(0,2,1)]
        
        raw_emb, Ax = STmodel(raw_x,A_list)

        if(self.PatchFSL_cfg['patch_encoder'] == 'raw'):
            input_features = [raw_emb]
        elif(self.PatchFSL_cfg['patch_encoder'] == 'pattern'):
            input_features = [raw_emb, pattern]
        input_features = torch.cat(input_features, dim = 2)

        out = FCmodel(input_features)
        
        # unnorm
        out = unnorm(out, means, stds)
        
        #return out,y, Ax
        if self.PatchFSL_cfg['patch_encoder'] == 'pattern':
            return out, y, Ax, pattern
        else:
            return out, y, Ax, None

class STRep(nn.Module):
    """
    Reptile-based Few-shot learning architecture for STGNN
    """
    def __init__(self, data_args, task_args, model_args,PatchFSL_cfg, model='GWN'):
        super(STRep, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.PatchFSL_cfg = PatchFSL_cfg

        self.update_lr = model_args['STnet']['update_lr']
        # self.update_lr = 0.0005
        self.meta_lr = model_args['STnet']['meta_lr']
        self.update_step = model_args['STnet']['update_step']
        # update_step_test is not used. It is replaced by target_epochs in main.
        self.task_num = task_args['maml']['task_num']
        self.model_name = model
        self.device = PatchFSL_cfg['device']
        self.current_epoch = 0

        self.model = PatchFSL(data_args, model_args, task_args,PatchFSL_cfg).to(self.device)
        for name, params in self.model.named_parameters():
            print("{} : {}, require_grads : {}".format(name, params.shape,params.requires_grad))
            
        # print(self.model)
        print("model params: ", count_parameters(self.model))

        self.meta_optim = optim.AdamW(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        # self.meta_optim = torch.optim.SGD(self.model.parameters(), lr=self.update_lr, momentum=0.9)
        self.loss_criterion = nn.MSELoss(reduction='mean')

        self.register_buffer('prior_prototypes', None)
        self.register_buffer('global_prior', None)
    
    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (update_step) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.update_step)) * (
                1.0 / self.update_step)
        decay_rate = 1.0 / self.update_step / self.task_args['maml']['train_epochs']
        # print("decay_rate : {}".format(decay_rate))
        min_value_for_non_final_losses = 0.03 / self.update_step
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            # print("each step : {}, {}".format(self.current_epoch * decay_rate, min_value_for_non_final_losses))
            loss_weights[i] = curr_value
        # print("loss_weights : {}".format(loss_weights))

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.update_step - 1) * decay_rate),
            1.0 - ((self.update_step - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights
    
    def graph_reconstruction_loss(self, meta_graph, adj_graph):
        adj_graph = adj_graph.unsqueeze(0).float()
        for i in range(meta_graph.shape[0]):
            if i == 0:
                matrix = adj_graph
            else:
                matrix = torch.cat((matrix, adj_graph), 0)
        criteria = nn.MSELoss()
        loss = criteria(meta_graph, matrix.float())
        return loss
    
    def calculate_loss(self, out, y, meta_graph, matrix, stage='target', graph_loss=True, loss_lambda=0):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.graph_reconstruction_loss(meta_graph, matrix)
            else:
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.loss_criterion(meta_graph, matrix.float())
            loss = loss_predict + loss_lambda * loss_reconsturct
        else:
            loss = self.loss_criterion(out, y)

        return loss
    
    
    
    def meta_train_revise(self, data_spt, matrix_spt, data_qry, matrix_qry):
        loss_weights = self.get_per_step_loss_importance_vector().detach()
        init_params = deepcopy(list(self.model.parameters()))
        total_mae, total_mse, total_rmse, total_mape = [], [], [], []
        task_losses = []
        task_patterns = [] 

        for i in range(self.task_num):
            for idx, (init_param, model_param) in enumerate(zip(init_params, self.model.parameters())):
                model_param.data = init_param
            task_loss = 0
            for k in range(self.update_step):
                # ================= Support 阶段 =================
                batch_size, node_num, seq_len, _ = data_spt[i].x.shape
                if self.model_name == 'GWN':
                    A = matrix_spt[i]
                    A = A.unsqueeze(0).float()
                    for batch_i in range(batch_size):
                        if batch_i == 0:
                            A_gnd = A
                        else:
                            A_gnd = torch.cat((A_gnd, A), 0)
                    A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)
                else:
                    A_gnd = matrix_spt[i].to(self.device)

                # 提取 Support 特征
                out, y, meta_graph, _ = self.model(data_spt[i], A_gnd)
                
                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss = self.loss_criterion(out, y)
                else:
                    loss = self.calculate_loss(out, y, meta_graph, matrix_spt[i], 'source', graph_loss=False)
                
                # 第一次反向传播（此时 Support 相关的计算图会被自动释放）
                grad = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
                grad = list(grad)
                for idx,a in enumerate(grad):
                    if(a==None):
                        grad[idx] = torch.tensor(0.0)
                
                for idx, (gra, model_param) in enumerate(zip(grad, self.model.parameters())):
                    model_param.data = model_param.data - self.update_lr * gra
                    
                del grad
                
                # ================= Query 阶段 =================
                batch_size, node_num, seq_len, _ = data_qry[i].x.shape
                if self.model_name == 'GWN':
                    A = matrix_qry[i]
                    A = A.unsqueeze(0).float()
                    for batch_i in range(batch_size):
                        if batch_i == 0:
                            A_gnd = A
                        else:
                            A_gnd = torch.cat((A_gnd, A), 0)
                    A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)
                else:
                    A_gnd = matrix_qry[i].to(self.device)
                    
                # ⚠️ 必须直接覆盖旧的 out, y, meta_graph 变量，否则后续 loss_q 会算错图
                out, y, meta_graph, pattern_qry = self.model(data_qry[i], A_gnd)
                
                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss_q = self.loss_criterion(out, y)
                else:
                    loss_q = self.calculate_loss(out, y, meta_graph, matrix_spt[i], 'source', graph_loss=False)
                
                task_loss += loss_weights[k] * loss_q
                del loss_q
                
                if k == self.update_step - 1:
                    # 仅截取特征并强行打断梯度
                    if pattern_qry is not None:
                        task_patterns.append(pattern_qry.detach().view(-1, pattern_qry.size(-1)))
                    MSE,RMSE,MAE,MAPE = calc_metric(out, y)
                    total_mse.append(MSE.cpu().detach().numpy())
                    total_rmse.append(RMSE.cpu().detach().numpy())
                    total_mae.append(MAE.cpu().detach().numpy())
                    total_mape.append(MAPE.cpu().detach().numpy())
            
            task_losses.append(task_loss)

        # 外层元梯度更新
        model_loss = torch.sum(torch.stack(task_losses))
        grad = torch.autograd.grad(model_loss, self.model.parameters(), allow_unused=True)
        grad = list(grad)
        for idx,a in enumerate(grad):
            if(a==None):
                grad[idx] = torch.tensor(0.0)
        for init_param, model_param, gra in zip(init_params, self.model.parameters(),grad):
            model_param.data = init_param - self.update_lr * gra

        # ================= 全局簇分布先验 (严格阻断跨 Epoch 梯度) =================
        if len(task_patterns) > 0:
            features = torch.cat(task_patterns, dim=0)
            K = self.PatchFSL_cfg.get('K', 10)
            
            if self.prior_prototypes is None or self.prior_prototypes.size(-1) != features.size(-1):
                idx = torch.randperm(features.size(0), device=self.device)[:K]
                self.prior_prototypes = features[idx].clone().detach()
                self.global_prior = (torch.ones(K) / K).to(self.device)
                
            dists = torch.cdist(features, self.prior_prototypes)
            labels = torch.argmin(dists, dim=1)
            new_protos = []
            probs = []
            for c in range(K):
                mask = labels == c
                probs.append(mask.float().mean())
                if mask.sum() > 0:
                    new_protos.append(features[mask].mean(dim=0))
                else:
                    new_protos.append(self.prior_prototypes[c])
                    
            new_protos = torch.stack(new_protos).detach()
            new_prior = torch.stack(probs).detach()
            
            momentum = 0.9
            self.prior_prototypes = (momentum * self.prior_prototypes + (1 - momentum) * new_protos).detach()
            self.global_prior = (momentum * self.global_prior + (1 - momentum) * new_prior).detach()
        # ========================================================================
            
        self.current_epoch+=1
        return model_loss.detach().cpu().numpy(), np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae), np.mean(total_mape)
    
    def forward(self, data, matrix):
        out, meta_graph = self.model(data, matrix)
        return out, meta_graph

    def train_batch(self, start,end,source_dataset,loss_fn,opts):
        total_loss = []
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []

        total_finetune_epochs = self.task_args['maml'].get('finetune_epochs', 10) 
        progress = min(self.current_epoch / total_finetune_epochs, 1.0)
        
        for idx in range(start,end):
            data_i, A = source_dataset[idx]
            B, N, D, L = data_i.x.shape
            A = A.unsqueeze(0).float()
            for i in range(B):
                if i == 0:
                    A_gnd = A
                else:
                    A_gnd = torch.cat((A_gnd, A), 0)
            A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)

            out, y, Ax, pattern = self.model(data_i, A_gnd)
            loss = loss_fn(out, y)

            # ================= 新增：目标域 KL 散度分配对齐 =================
            if pattern is not None and self.prior_prototypes is not None:
                feat = pattern.view(-1, pattern.size(-1))
                dists = torch.cdist(feat, self.prior_prototypes)
                
                # 计算目标城市当前 batch 的簇分配对数概率 (log_softmax)
                #target_log_probs = F.log_softmax(-dists, dim=1).mean(dim=0)

                # 引入温度系数 T=2.0 软化分布，给予模型更多在目标城市的弹性
                T = 1.0 + 4.0 * progress
                target_log_probs = F.log_softmax(-dists / T, dim=1).mean(dim=0)

                # 将 global_prior 也进行一定的平滑处理 (防极端 0 概率)
                smoothed_prior = (self.global_prior + 1e-5) / (self.global_prior + 1e-5).sum()
                
                # 约束目标城市的交通模式使用频率，向多源城市的全局先验逼近
                # PyTorch KLDivLoss: input 需为 log-prob, target 需为 prob
                kl_loss = F.kl_div(target_log_probs, smoothed_prior, reduction='sum')
                alpha = 0.1 * (1.0 - 0.9 * progress)                
                #alpha = 0.05 # 对齐正则化系数
                loss = loss + alpha * kl_loss
            # ===============================================================

            for opt in opts:
                opt.zero_grad()
            loss.backward()
            for opt in opts:
                opt.step()
        
            MSE,RMSE,MAE,MAPE = calc_metric(out, y)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())
            total_loss.append(loss.item())
        return total_mse,total_rmse, total_mae, total_mape, total_loss

    def test_batch(self,start,end,source_dataset,stage = "test"):
        total_loss = []
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        
        with torch.no_grad():
            for idx in range(start,end):
                data_i, A = source_dataset[idx]
                B, N, D, L = data_i.x.shape
                A = A.unsqueeze(0).float()
                for i in range(B):
                    if i == 0:
                        A_gnd = A
                    else:
                        A_gnd = torch.cat((A_gnd, A), 0)
                A_gnd = torch.tensor(A_gnd,dtype=torch.float32).to(self.device)
                # A_gnd = A
                
                # activate .eval()
                out,y,Ax,_ = self.model(data_i, A_gnd, stage='test')
                
                # print 12 horizons
                MSE,RMSE,MAE,MAPE = calc_metric(out, y, stage='test')
                total_mse.append(MSE.cpu().detach().numpy())
                total_rmse.append(RMSE.cpu().detach().numpy())
                total_mae.append(MAE.cpu().detach().numpy())
                total_mape.append(MAPE.cpu().detach().numpy())
        return total_mse,total_rmse, total_mae, total_mape, total_loss


    def finetuning(self, finetune_dataset, test_dataset, target_epochs):
        """
        finetunning stage in MAML
        """
        curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")
        model_path = Path('./save/meta_model/{}/{}/'.format(self.PatchFSL_cfg['data_list'],curr_time))
        
        if(not os.path.exists(model_path)):
            os.makedirs(model_path)
        print("Finetuned model saved in {}".format(model_path))
        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)

        best_loss = 9999999999999.0
        best_model = None
        print("[INFO] Enter finetune phase")

        for i in range(target_epochs):
            length = finetune_dataset.__len__()
            # length=40
            print('----------------------')
            time_1 = time.time()

            total_mse,total_rmse, total_mae, total_mape, total_loss = self.train_batch(0,length, finetune_dataset,self.loss_criterion,[optimizer])
            print('Epochs {}/{}'.format(i,target_epochs))
            print('in training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, normed MSE : {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape),np.mean(total_loss)))
            
            mae_loss = np.mean(total_mae)
            if(mae_loss < best_loss):
                torch.save(self.model.state_dict(), model_path / 'finetuned_bestmodel.pt')
                best_model = copy.deepcopy(self.model)
                best_loss = mae_loss
                print('Best model. Saved.')
            print('this epoch costs {:.5}s'.format(time.time()-time_1))

        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
        self.model = copy.deepcopy(best_model)
        
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.test_batch(0,length, test_dataset,"test")
        for i in range(self.task_args['maml']['pred_num']):
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            for j in range(len(total_mse_horizon)):
                
                total_mse.append(total_mse_horizon[j][i])
                total_rmse.append(total_rmse_horizon[j][i])
                total_mae.append(total_mae_horizon[j][i])
                total_mape.append(total_mape_horizon[j][i])
                
            print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i,np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))
