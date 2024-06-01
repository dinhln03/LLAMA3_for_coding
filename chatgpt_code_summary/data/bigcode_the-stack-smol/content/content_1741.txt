"""
  @Author       : liujianhan
  @Date         : 2018/5/15 上午10:48
  @Project      : KGE
  @FileName     : service.py
  @Description  : 服务接口模块
"""
import codecs
import json
import os
import time
from typing import Dict

import torch
from dotmap import DotMap

from .core.predict import get_entity_relation_with_id
from .layer.model import KGEModel

kge_model, entity2id, id2entity, relation2id, all_true_triples, args = None, None, None, None, None, None


def load_model(model_path: str) -> None:
    """
    模型加载
    @param model_path: 模型文件夹路径
    @return:
    """
    global kge_model, entity2id, id2entity, relation2id, all_true_triples, args
    args = DotMap(json.load(codecs.open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8')))
    entity2id, id2entity, relation2id, id2relation, all_true_triples = get_entity_relation_with_id(args.data_path)

    kge_model = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    if args.cuda:
        kge_model = kge_model.cuda()

    checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
    kge_model.load_state_dict(checkpoint['model_state_dict'])


def inference(target_triple: str) -> Dict:
    """
    推理函数
    @param target_triple: 目标需预测三元组：'头实体 关系 尾实体'
    @return: 头尾实体的10个预测结果
    """
    if kge_model is None:
        return {'预测结果': '提醒：模型未加载'}
    try:
        target_triple = target_triple.split()
        head = entity2id[target_triple[0]]
        tail = entity2id[target_triple[2]]
        relation = relation2id[target_triple[1]]
        target_triple = [(head, relation, tail)]
    except KeyError as e:
        return {'预测结果': f'实体或者关系 <{e}> 不存在，请确保输入的实体或者关系已存在。'}

    prediction = kge_model.test_step(kge_model, target_triple, all_true_triples, args, True)
    head_entity_prediction = [id2entity[str(idx)] for idx in prediction['head_predict']]
    tail_entity_prediction = [id2entity[str(idx)] for idx in prediction['tail_predict']]
    result = {'头实体预测结果': head_entity_prediction, '尾实体预测结果': tail_entity_prediction}

    return result


if __name__ == '__main__':
    t1 = time.time()
    load_model('data_path/model/DistMult_cn_military_300k_10')
    test_cases = [
        '摩耶号/Maya巡洋舰 建造时间 1928年',
        '1949年2月28日 星座 双鱼座'
    ]
    t2 = time.time()
    res = inference(test_cases[0])
    print(f'模型加载耗时： {t2 - t1: .3}s')
    print(f'推理耗时： {time.time() - t2: .3}s')
    print(res)
