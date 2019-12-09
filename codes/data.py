import os
import torch
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict
from torch.utils.data import Dataset


class BatchType(Enum):
    HEAD_BATCH = 0
    TAIL_BATCH = 1
    SINGLE = 2


class ModeType(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class DataReader(object):
    def __init__(self, data_path: str):
        entity_dict_path = os.path.join(data_path, 'entities.dict')
        relation_dict_path = os.path.join(data_path, 'relations.dict')
        train_data_path = os.path.join(data_path, 'train.txt')
        valid_data_path = os.path.join(data_path, 'valid.txt')
        test_data_path = os.path.join(data_path, 'test.txt')

        self.entity_dict = self.read_dict(entity_dict_path)
        self.relation_dict = self.read_dict(relation_dict_path)

        self.train_data = self.read_data(train_data_path, self.entity_dict, self.relation_dict)
        self.valid_data = self.read_data(valid_data_path, self.entity_dict, self.relation_dict)
        self.test_data = self.read_data(test_data_path, self.entity_dict, self.relation_dict)

    def read_dict(self, dict_path: str):
        """
        Read entity / relation dict.
        Format: dict({id: entity / relation})
        """

        element_dict = {}
        with open(dict_path, 'r') as f:
            for line in f:
                id_, element = line.strip().split('\t')
                element_dict[element] = int(id_)

        return element_dict

    def read_data(self, data_path: str, entity_dict: Dict[str, int], relation_dict: Dict[str, int]):
        """
        Read train / valid / test data.
        """
        triples = []
        with open(data_path, 'r') as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                triples.append((entity_dict[head], relation_dict[relation], entity_dict[tail]))
        return triples


class TrainDataset(Dataset):
    def __init__(self, data_reader: DataReader, neg_size: int, batch_type: BatchType):
        """
        Dataset for training, inherits `torch.utils.data.Dataset`.
        Args:
            data_reader: DataReader,
            neg_size: int, negative sample size.
        """

        self.triples = data_reader.train_data
        self.len = len(self.triples)
        self.num_entity = len(data_reader.entity_dict)
        self.num_relation = len(data_reader.relation_dict)
        self.neg_size = neg_size
        self.batch_type = batch_type

        self.hr_map, self.tr_map, self.hr_freq, self.tr_freq = self.two_tuple_count()

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        """
        Returns a positive sample and `self.neg_size` negative samples.
        """
        pos_triple = self.triples[idx]
        head, rel, tail = pos_triple

        subsampling_weight = self.hr_freq[(head, rel)] + self.tr_freq[(tail, rel)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        neg_triples = []
        neg_size = 0

        while neg_size < self.neg_size:
            neg_triples_tmp = np.random.randint(self.num_entity, size=self.neg_size * 2)
            if self.batch_type == BatchType.HEAD_BATCH:
                mask = np.in1d(
                    neg_triples_tmp,
                    self.tr_map[(tail, rel)],
                    assume_unique=True,
                    invert=True
                )
            elif self.batch_type == BatchType.TAIL_BATCH:
                mask = np.in1d(
                    neg_triples_tmp,
                    self.hr_map[(head, rel)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Invalid BatchType: {}'.format(self.batch_type))

            neg_triples_tmp = neg_triples_tmp[mask]
            neg_triples.append(neg_triples_tmp)
            neg_size += neg_triples_tmp.size

        neg_triples = np.concatenate(neg_triples)[:self.neg_size]

        pos_triple = torch.LongTensor(pos_triple)
        neg_triples = torch.from_numpy(neg_triples)

        return pos_triple, neg_triples, subsampling_weight, self.batch_type

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        batch_type = data[0][3]
        return positive_sample, negative_sample, subsample_weight, batch_type

    def two_tuple_count(self):
        """
        Return two dict:
        dict({(h, r): [t1, t2, ...]}),
        dict({(t, r): [h1, h2, ...]}),
        """
        hr_map = {}
        hr_freq = {}
        tr_map = {}
        tr_freq = {}

        init_cnt = 3
        for head, rel, tail in self.triples:
            if (head, rel) not in hr_map.keys():
                hr_map[(head, rel)] = set()

            if (tail, rel) not in tr_map.keys():
                tr_map[(tail, rel)] = set()

            if (head, rel) not in hr_freq.keys():
                hr_freq[(head, rel)] = init_cnt

            if (tail, rel) not in tr_freq.keys():
                tr_freq[(tail, rel)] = init_cnt

            hr_map[(head, rel)].add(tail)
            tr_map[(tail, rel)].add(head)
            hr_freq[(head, rel)] += 1
            tr_freq[(tail, rel)] += 1

        for key in tr_map.keys():
            tr_map[key] = np.array(list(tr_map[key]))

        for key in hr_map.keys():
            hr_map[key] = np.array(list(hr_map[key]))

        return hr_map, tr_map, hr_freq, tr_freq


class TestDataset(Dataset):
    def __init__(self, data_reader: DataReader, mode: ModeType, batch_type: BatchType):
        self.triple_set = set(data_reader.train_data + data_reader.valid_data + data_reader.test_data)
        if mode == ModeType.VALID:
            self.triples = data_reader.valid_data
        elif mode == ModeType.TEST:
            self.triples = data_reader.test_data

        self.len = len(self.triples)

        self.num_entity = len(data_reader.entity_dict)
        self.num_relation = len(data_reader.relation_dict)

        self.mode = mode
        self.batch_type = batch_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.batch_type == BatchType.HEAD_BATCH:
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.num_entity)]
            tmp[head] = (0, head)
        elif self.batch_type == BatchType.TAIL_BATCH:
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.num_entity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch type {} not supported'.format(self.mode))

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.batch_type

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
