import random
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KGIRA import Recommender
from utils.evaluate import test
from utils.helper import early_stopping, init_logger

import collections
import pandas as pd
from logging import getLogger
from scipy.sparse import coo_matrix

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


def load_kg(filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data


def load_item_entities():
        kg_file = args.data_path + args.dataset + '/' + 'kg_final.txt'
        kg_data = load_kg(kg_file)
        item_entities = collections.defaultdict(set)
        item_relations = collections.defaultdict(set)
        for row in kg_data.loc[kg_data['h'] < n_items].iterrows():
            h, r, t = row[1]
            item_entities[h].add(t)
            item_relations[h].add(r)
        for row in kg_data.loc[kg_data['t'] < n_items].iterrows():
            h, r, t = row[1]
            item_entities[t].add(h)
            item_relations[t].add(r)
        for k,v in item_entities.items():
            item_entities[k] = list(v)
        for k,v in item_relations.items():
            item_relations[k] = list(v)
        return item_entities, item_relations


def construct_sim(sim_path):
        def jaccard_sim(mat, n_items):
            co_share = torch.matmul(mat,mat.t())
            diag = torch.diag(co_share, 0)
            diag_reshape = torch.reshape(diag,(diag.shape[0],1))
            expand = diag_reshape.expand(diag_reshape.shape[0],n_items)
            union = expand+diag
            result = (co_share/(union - co_share))
            return result
        if not os.path.isfile(sim_path+'/item_sim.t'):
            adj = torch.zeros(n_items, n_entities)
            for k,v in item_entities.items():
                for i in v:
                    adj[k][i] = 1.0
            item_sim_tensor = jaccard_sim(adj, n_items)
            item_sim_tensor = torch.nan_to_num(item_sim_tensor)
            torch.save(item_sim_tensor, sim_path+'/item_sim.t')
        item_sim_tensor = torch.load(sim_path+'/item_sim.t')
        return item_sim_tensor


def calculate_jaccard_similarity(graph, device='cpu'):
    row, col = zip(*graph.edges())
    data = [1] * len(row)
    adj_sparse = coo_matrix((data, (row, col)), shape=(n_entities, n_entities))
    adj_torch = torch.sparse.FloatTensor(
        torch.LongTensor([adj_sparse.row.tolist(), adj_sparse.col.tolist()]),
        torch.FloatTensor(adj_sparse.data),
        torch.Size(adj_sparse.shape)).to(device)
    co_share = torch.sparse.mm(adj_torch, adj_torch.t())
    diag = torch.sparse.sum(adj_torch, dim=1).to_dense()
    def jaccard_similarity(heads, tails):
        heads, tails = heads.to(device), tails.to(device)
        intersections = torch.tensor([co_share[head, tail].item() for head, tail in zip(heads, tails)], device=device)
        unions = diag[heads] + diag[tails] - intersections
        similarities = intersections / unions
        similarities[unions == 0] = 0
        return similarities
    return jaccard_similarity


if __name__ == '__main__':
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    log_fn = init_logger(args)
    logger = getLogger()

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """new"""
    item_entities, item_relations = load_item_entities()

    item_sim_tensor = construct_sim(args.data_path + args.dataset + '/')
    item_sim_tensor = item_sim_tensor.to(device)

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0], item_entities, item_sim_tensor).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    logger.info("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        with tqdm(total=len(train_cf)//args.batch_size) as pbar:
            while s + args.batch_size <= len(train_cf):
                batch = get_feed_dict(train_cf_pairs,
                                    s, s + args.batch_size,
                                    user_dict['train_user_set'])
                batch_loss, _, _, batch_cor = model(batch)

                batch_loss = batch_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                cor_loss += batch_cor
                s += args.batch_size
                pbar.update(1)

        train_e_t = time()

        if epoch >= 1:
            """testing"""
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            logger.info(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '-KGIN.ckpt')

        else:
            logger.info('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, loss.item(), cor_loss.item()))

    logger.info('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
