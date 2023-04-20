import numpy as np
import time
import torch
import torch.nn as nn
from copy import deepcopy
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize
import argparse
import torch.optim as optim
from utils import *
from data import Dataset
import random
from model_lenet import MultiLeNet
"""
Define task metrics, loss functions and model trainer here.
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

''' ===== multi task MGD trainer ==== '''
def multi_task_mgd_trainer(train_loader, test_loader, multi_task_model, device,
                           optimizer, scheduler, opt,
                           total_epoch=200, method='sumloss', alpha=0.5, seed=0,dataset_name=None):
    start_time = time.time()

    def graddrop(grads):
        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
        U = torch.rand_like(grads[:,0])
        M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
        g = (grads * M.float()).mean(1)
        return g

    def mgd(grads):
        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([
            grads_cpu[t] for t in range(grads.shape[-1])])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def pcgrad(grads, rng):
        grad_vec = grads.t()
        num_tasks = 2

        shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
        for i in range(num_tasks):
            task_indices = np.arange(num_tasks)
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T

        normalized_grad_vec = grad_vec / (
            grad_vec.norm(dim=1, keepdim=True) + 1e-8
        )  # num_tasks x dim
        modified_grad_vec = deepcopy(grad_vec)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[
                task_indices
            ]  # num_tasks x dim
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(
                dim=1, keepdim=True
            )  # num_tasks x dim
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
        g = modified_grad_vec.mean(dim=0)
        return g

    def cagrad(grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient
        num_tasks = 2
        x_start = np.ones(num_tasks) / num_tasks
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale== 0:
            return g
        elif rescale == 1:
            return g / (1+alpha**2), ww.detach().cpu().numpy()
        else:
            return g / (1 + alpha)
    def ada_cagrad(loss,grads, alpha=0.5, rescale=1, sigma = 0.2, kappa = 0.3):
        GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient
        num_tasks = 2
        x_start = np.ones(num_tasks) / num_tasks
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale== 0:
            return g, ww
        elif rescale == 1:
            return g / (1+alpha**2)
        else:
            return g / (1 + alpha)
    def grad2vec(m, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        for mm in m.shared_modules():
            for p in mm.parameters():
                grad = p.grad
                if grad is not None:
                    grad_cur = grad.data.detach().clone()
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[:cnt + 1])
                    grads[beg:en, task].copy_(grad_cur.data.view(-1))
                cnt += 1

    def overwrite_grad(m, newgrad, grad_dims):
        newgrad = newgrad * 2 # to match the sum loss
        cnt = 0
        for mm in m.shared_modules():
            for param in mm.parameters():
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(param.data.size())
                param.grad = this_grad.data.clone()
                cnt += 1
    rng = np.random.default_rng()
    grad_dims = []
    for mm in multi_task_model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), 2).cuda()

    train_batch = len(train_loader)
    test_batch = len(test_loader)
    acc_task1_all, acc_task2_all, loss_task1_all, loss_task2_all = [], [], [], []
    for index in range(total_epoch):
        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        total = 0.0
        losses = 0.0
        sigma = np.random.rand(1)
        kappa = np.random.rand(1)
        for k in range(train_batch):
            img,ys = train_dataset.next()
            img, ys = img.to(device), ys.long().to(device)
            bs = len(ys)
            task1, task2 = multi_task_model(img)
            train_loss_tmp = [nn.CrossEntropyLoss()(task1,ys[:, 0]),nn.CrossEntropyLoss()(task2,ys[:, 1])]
            prev_loss = np.array([train_loss_tmp[0].detach().cpu().numpy(),train_loss_tmp[1].detach().cpu().numpy()])
            optimizer.zero_grad()
            if method == "graddrop":
                for i in range(2):
                    if i < 2:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "mgd":
                for i in range(2):
                    if i < 2:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = mgd(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "pcgrad":
                for i in range(2):
                    if i < 2:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = pcgrad(grads, rng)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "cagrad":
                for i in range(2):
                    if i < 2:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g,w = cagrad(grads, alpha, rescale=1)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "ada_cagrad":
                for i in range(2):
                    if i < 2:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g,w = cagrad(grads, alpha, rescale=1)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
                cur_loss = np.array([train_loss_tmp[0].detach().cpu().numpy(),train_loss_tmp[1].detach().cpu().numpy()])
                g_norm = g.unsqueeze(0)
                if (w*cur_loss).sum() <=  (w*prev_loss).sum() + sigma * optimizer.param_groups[0]['lr']*((((g_norm.mm(g_norm.t())).norm())**2).detach().cpu().numpy()):
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
                else:
                    optimizer.param_groups[0]['lr'] = kappa* optimizer.param_groups[0]['lr']
                    print("Update learning rate: ",optimizer.param_groups[0]['lr'])
            # loss
            losses_batch = [train_loss_tmp[0].detach().cpu().tolist(),train_loss_tmp[1].detach().cpu().tolist()]
            losses += bs * np.array(losses_batch)
            total += bs
        l1_train = losses[0] / total
        l2_train = losses[1] / total
        #print('Epoch: {:04d}, Loss_train_task1: {:.4f}, Loss_train_task2: {:.4f}'.format(index,losses[0] / total,losses[1] / total))
        # evaluating test data
        multi_task_model.eval()
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            task1_correct, task2_correct = 0.0, 0.0
            total = 0.0
            for k in range(test_batch):
                img,ys = test_dataset.next()
                img, ys = img.to(device), ys.long().to(device)
                task1, task2 = multi_task_model(img)
                bs = len(ys)

                # acc
                pred1 = task1.data.max(1)[1]  # first column has actual prob.
                pred2 = task2.data.max(1)[1]  # first column has actual prob.
                task1_correct += pred1.eq(ys[:, 0]).sum()
                task2_correct += pred2.eq(ys[:, 1]).sum()
                total += bs

            print('Epoch: {:04d}, Acc_val_task1: {:.4f}, Acc_val_task2: {:.4f}, Loss_train_task1: {:.4f}, Loss_train_task2: {:.4f}'.format(index,task1_correct.cpu().item() / total,task2_correct.cpu().item() / total,l1_train,l2_train))
            acc_task1_all.append(task1_correct.cpu().item() / total)
            acc_task2_all.append(task2_correct.cpu().item() / total)
            loss_task1_all.append(l1_train)
            loss_task2_all.append(l2_train)
        #scheduler.step()

        if "cagrad" in method:
            torch.save(multi_task_model.state_dict(), f"models/{dataset_name}_{method}-{opt.weight}-{alpha}-{seed}.pt")
        else:
            torch.save(multi_task_model.state_dict(), f"models/{dataset_name}_{method}-{opt.weight}-{seed}.pt")
    avg_acc1 = np.mean(np.array(acc_task1_all))
    avg_acc2 = np.mean(np.array(acc_task2_all))
    avg_acc = (avg_acc1 + avg_acc2)/2
    avg_loss1 = np.mean(np.array(loss_task1_all))
    avg_loss2 = np.mean(np.array(loss_task2_all))
    avg_loss = (avg_loss1 + avg_loss2)/2
    print('AVG_acc: {:.4f}, AVG_acc_val_task1: {:.4f}, AVG_acc_val_task2: {:.4f},AVG_loss: {:.4f}, AVG_loss_train_task1: {:.4f}, AVG_loss_train_task2: {:.4f}'.format(avg_acc,avg_acc1,avg_acc2,avg_loss,avg_loss1,avg_loss2))
    end_time = time.time()
    print("Training time: ", end_time-start_time)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-task: Split')
    parser.add_argument('--dataname', default='multi_mnist', type=str, help='multi_mnist, multi_fashion, multi_fashion_and_mnist')
    parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
    parser.add_argument('--method', default='cagrad', type=str, help='optimization method')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
    parser.add_argument('--alpha', default=0.2, type=float, help='the alpha')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
    parser.add_argument('--seed', default=0, type=int, help='the seed')
    parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
    opt = parser.parse_args()
    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # define model, optimiser and scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = MultiLeNet(n_tasks = 2).to(device)
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=0.001,weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print('Parameter Space: ABS: {:.1f}'.format(count_parameters(SegNet_MTAN)))

    # define dataset
    path = opt.dataroot+opt.dataname+".pickle"
    val_size = -1
    bs = opt.bs
    data = Dataset(path, val_size=val_size)
    if val_size > 0:
        train_set, val_set, test_set = data.get_datasets()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=bs, shuffle=False, num_workers=2
        )
    else:
        train_set, val_set = data.get_datasets()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=False, num_workers=2
        )
    # Train and evaluate multi-task network
    multi_task_mgd_trainer(train_loader,
                    val_loader,
                    SegNet_MTAN,
                    device,
                    optimizer,
                    scheduler,
                    opt,
                    50,
                    opt.method,
                    opt.alpha, opt.seed,opt.dataname)