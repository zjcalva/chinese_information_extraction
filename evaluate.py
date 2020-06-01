import numpy as np
import torch
from collections import defaultdict
import pdb

def evaluate(gold_entities, pred_entities):
    prec_all_num, prec_num, recall_all_num, recall_num = 0, 0, 0, 0
    for g_ets, p_ets in zip(gold_entities, pred_entities):
        recall_all_num += len(g_ets)
        prec_all_num += len(p_ets)

        for et in g_ets:
            if et in p_ets:
                recall_num += 1
        
        for et in p_ets:
            if et in g_ets:
                prec_num += 1
    
    return prec_all_num, prec_num, recall_all_num, recall_num

def evaluate_relation(gold_relations,pre_relations):
    pred_all, pred, recall_all, recall = 0, 0, 0, 0
    prec_all_num,prec_num,recall_all_num,recall_num=0,0,0,0
    for g_rels,p_rels in zip(gold_relations,pre_relations):
        for p_rel in p_rels:
            if p_rels[p_rel]!='None':
                prec_all_num+=1
                if (p_rel[0],p_rel[1]) in g_rels:
                    if g_rels[(p_rel[0],p_rel[1])]!='None':
                        prec_num+=1
                elif (p_rel[1],p_rel[0]) in g_rels:
                    if g_rels[(p_rel[1],p_rel[0])]!='None':
                        prec_num+=1
        for g_rel in g_rels:
            if g_rels[g_rel]!='None':
                recall_all_num+=1
                if (g_rel[0],g_rel[1]) in p_rels:
                    if p_rels[(g_rel[0],g_rel[1])]!='None':
                        recall_num+=1
                elif (g_rel[1],g_rel[0]) in p_rels:
                    if p_rels[(g_rel[1],g_rel[0])]!='None':
                        recall_num+=1
    return prec_all_num,prec_num,recall_all_num,recall_num

def get_f1(model, batch_zip, config):
    pred_all, pred, recall_all, recall = 0, 0, 0, 0

    num_invalids = 0
    num_words = 0

    for token_batch, gaz_batch, label_batch, action_batch in batch_zip:
        token_batch_var = torch.LongTensor(np.array(token_batch))
        #pos_batch_var = torch.LongTensor(np.array(pos_batch))
        if config.if_gpu:
            token_batch_var = token_batch_var.cuda()
            #pos_batch_var = pos_batch_var.cuda()
        
        sent_len, batch_size = token_batch_var.size()
        num_words += sent_len * batch_size

        with torch.no_grad():
            model.eval()
            pred_entities = model.predict(token_batch_var,gaz_batch)
            # print (label_batch,pred_entities)
            p_a, p, r_a, r = evaluate(label_batch, pred_entities)
        pred_all += p_a
        pred += p
        recall_all += r_a
        recall += r

    if pred_all == 0:
        p = 0
    else:
        p = pred / pred_all

    r = recall / recall_all
    if p == 0 or r == 0:
        f1 =0
    else:
        f1 = 2.0 / ((1.0 / p) + (1.0 / r)) 
    print( "Entity Precision {0}, Recall {1}, F1 {2}".format(p, r , f1) )


    print( "Number of invalid mentions {}".format(num_invalids) )
    print( "Number of word {}".format(num_words) )
    return  f1

def get_action_acc(acc_num, total_num):
    acc = 0
    total = 0
    for a, t in zip(acc_num, total_num):
        acc += a
        total += t
    return acc * 1.0 / total
