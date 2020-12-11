import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from MFGNet import MFGNet, GNN
from test_modelnet40 import my_plot_pc, plot_pc
from data import ModelNet40
from util import npmat2euler, IOStream
from main import seed_everything


def test_one_epoch(args, net, test_loader):
    net.eval()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    for src, target, R, t, euler in tqdm(test_loader):

        src = src.cuda()
        target = target.cuda()
        R = R.cuda()
        t = t.cuda()

        R_pred, t_pred, *_ = net(src, target)


        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return r_rmse, r_mae, t_rmse, t_mae

if __name__ == '__main__':
    arg_bool = lambda x: x.lower() in ['true', 't', '1']
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='MFGNet_preweight', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--num_iter', type=int, default=4, metavar='N',
                        help='Number of iteration inside the network')
    parser.add_argument('--emb_nn', type=str, default='GNN', metavar='N',
                        help='Feature extraction method. [GNN]')
    parser.add_argument('--emb_dims', type=int, default=64, metavar='N',
                        help='Dimension of embeddings.')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--unseen', type=arg_bool, default='False',
                        help='Test on unseen categories')
    parser.add_argument('--gaussian_noise', type=arg_bool, default='True',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--alpha', type=float, default=0.75, metavar='N',
                        help='Fraction of points when sampling partial point cloud')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')  # 旋转角度在pi/4范围内

    args = parser.parse_args()

    seed_everything(42)
    loader = ModelNet40(num_points=1024, partition='test', alpha=args.alpha, gaussian_noise=args.gaussian_noise, unseen=args.unseen,
                   factor=args.factor)
    # loader.seed_everything(42)
    test_loader = DataLoader(
        loader,
        batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=8)

    model = MFGNet(GNN(args.emb_dims), args).cuda()
    device = torch.device('cuda')
    model.load_state_dict(torch.load('model_gaussian.t7'))
    rot_RMSE = []
    rot_MAE = []
    trans_RMSE = []
    trans_MAE = []
    # start = time.time()
    for i in range(50):
        start = time.time()
        test_stats = test_one_epoch(args, model, test_loader)
        rot_RMSE.append(test_stats[0])
        rot_MAE.append(test_stats[1])
        trans_RMSE.append(test_stats[2])
        trans_MAE.append(test_stats[3])
        print('TEST ,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % test_stats)
        end = time.time()
        print('average predict time : point clouds: %.4f' % ((end - start)/2468))
    print('Mean TEST,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % (np.mean(rot_RMSE),np.mean(rot_MAE),np.mean(trans_RMSE),np.mean(trans_MAE)))
