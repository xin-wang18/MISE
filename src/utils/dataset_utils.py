import torch
from torch.utils.data import Dataset
import numpy as np
import random

class MAMLDataset(Dataset):
    def __init__(self, opt, features, callback_info, mode, **kwargs):
        self.batchsz = 10
        # self.k_shot = 3
        # self.k_query = 15
        self.k_shot = opt.shot
        self.k_query = opt.query
        self.max_seq_len = opt.max_seq_len
        self.mode = mode
        if self.mode == 'dev':
            self.batchsz = 10
        data2features,data2features_callback_info = self.load_features(features, callback_info[0])
        # for item in data2features.items():
        #     print(item[1][0].__dict__)
        self.data=[]
        self.callback_info=[]
        for i, (k, v) in enumerate(data2features.items()):
            self.data.append(v)
        for i, (k, v) in enumerate(data2features_callback_info.items()):
            self.callback_info.append(v)      
        self.cls_num = len(self.data)
        self.create_batch(self.batchsz)
        # for data in self.data:
        #     print(data)
        #     print(len(data))
        # for callback_info in self.callback_info:
        #     print(callback_info)
        #     print(len(callback_info))
        # # print(data2features.keys())
        # # print(len(features))
        # exit()
        # self.nums = len(features)

        # self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        # self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        # self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        # self.labels = None
        # self.start_ids, self.end_ids = None, None
        # self.ent_type = None
        # self.pseudo = None
        # if mode == 'train':
        #     self.pseudo = [torch.tensor(example.pseudo).long() for example in features]
        #     if opt.task_type == 'crf':
        #         self.labels = [torch.tensor(example.labels) for example in features]
        #     else:
        #         self.start_ids = [torch.tensor(example.start_ids).long() for example in features]
        #         self.end_ids = [torch.tensor(example.end_ids).long() for example in features]

        # if kwargs.pop('use_type_embed', False):
        #     self.ent_type = [torch.tensor(example.ent_type) for example in features]

    def load_features(self,features,callback_info):
        dictdates={}
        dictdates_callback_info={}
        for example,example_callback_info in zip(features,callback_info):
            time=example.time
            if time in dictdates.keys():
                dictdates[time].append(example)
                dictdates_callback_info[time].append(example_callback_info)
                # print(example.__dict__)
                # print(example_callback_info)
                # exit()
            else:
                dictdates[time] = [example]
                dictdates_callback_info[time] = [example_callback_info]
                # print(example.__dict__)
                # print(example_callback_info)
                # exit()
        return dictdates,dictdates_callback_info
    
    def create_batch(self,batchsz):
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.query_x_callback_info_batch = [] #query callback label batch
        for b in range(batchsz):
            # selected_cls = np.random.choice(self.cls_num,1,False)
            selected_cls = np.random.randint(self.cls_num) # random select a time period
            # print(selected_cls)
            # exit()
            support_x = []
            query_x = []
            query_x_callback_info = []
            selected_tweet_idx = np.random.choice(len(self.data[selected_cls]),self.k_shot+self.k_query,False) # random select shot number+query number sample in this time period
            np.random.shuffle(selected_tweet_idx)
            # print(selected_tweet_idx)
            indexDtrain = np.array(selected_tweet_idx[:self.k_shot])
            indexDtest = np.array(selected_tweet_idx[self.k_shot:])
            support_x = np.array(self.data[selected_cls])[indexDtrain].tolist()
            query_x = np.array(self.data[selected_cls])[indexDtest].tolist()
            query_x_callback_info = np.array(self.callback_info[selected_cls])[indexDtest].tolist()
            # support_x.append(np.array(self.data[selected_cls])[indexDtrain].tolist())
            # query_x.append(np.array(self.data[selected_cls])[indexDtest].tolist())
            # random.shuffle(support_x)
            # random.shuffle(query_x)
            # print(support_x)
            # print(query_x)
            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)
            self.query_x_callback_info_batch.append(query_x_callback_info)

        # exit()
   

    def __len__(self):
        return self.batchsz

    def __getitem__(self, index):
        # print(len(self.support_x_batch))
        # print(self.query_x_batch[index][1].__dict__)
        # print(self.query_x_callback_info_batch[index][1])
        # exit()
        support_token_ids = torch.LongTensor(self.k_shot,self.max_seq_len)
        support_attention_masks = torch.LongTensor(self.k_shot,self.max_seq_len)
        support_token_type_ids = torch.LongTensor(self.k_shot,self.max_seq_len)
        support_labels = torch.LongTensor(self.k_shot,self.max_seq_len)
        support_pseudo = torch.LongTensor(self.k_shot)
        for i, example in enumerate(self.support_x_batch[index]):
            support_token_ids[i] = torch.tensor(example.token_ids)
            support_attention_masks[i] = torch.tensor(example.attention_masks)
            support_token_type_ids[i] = torch.tensor(example.token_type_ids)
            support_labels[i] = torch.tensor(example.labels)
            support_pseudo[i] = torch.tensor(example.pseudo)
        # print(support_token_ids)
        # token_ids = [torch.tensor(example.token_ids).long() for example in self.support_x_batch[index]]
        # print(token_ids)
        # print(support_attention_masks)
        # attention_masks = [torch.tensor(example.attention_masks).long() for example in self.support_x_batch[index]]
        # print(attention_masks)
        # print(support_token_type_ids)
        # token_type_ids = [torch.tensor(example.token_type_ids).long() for example in self.support_x_batch[index]]
        # print(token_type_ids)
        # print(support_labels)
        # labels = [torch.tensor(example.labels).long() for example in self.support_x_batch[index]]
        # print(labels)
        # print(support_pseudo)
        # pseudo = [torch.tensor(example.pseudo).long() for example in self.support_x_batch[index]]
        # print(pseudo)
        # exit()
        query_token_ids = torch.LongTensor(self.k_query,self.max_seq_len)
        query_attention_masks = torch.LongTensor(self.k_query,self.max_seq_len)
        query_token_type_ids = torch.LongTensor(self.k_query,self.max_seq_len)
        query_labels = torch.LongTensor(self.k_query,self.max_seq_len)
        query_pseudo = torch.LongTensor(self.k_query)
        for i, example in enumerate(self.query_x_batch[index]):
            query_token_ids[i] = torch.tensor(example.token_ids)
            query_attention_masks[i] = torch.tensor(example.attention_masks)
            query_token_type_ids[i] = torch.tensor(example.token_type_ids)
            query_labels[i] = torch.tensor(example.labels)
            query_pseudo[i] = torch.tensor(example.pseudo)
        # print(query_token_ids)
        # token_ids = [torch.tensor(example.token_ids).long() for example in self.query_x_batch[index]]
        # print(token_ids)
        # print(query_attention_masks)
        # attention_masks = [torch.tensor(example.attention_masks).long() for example in self.query_x_batch[index]]
        # print(attention_masks)
        # print(query_token_type_ids)
        # token_type_ids = [torch.tensor(example.token_type_ids).long() for example in self.query_x_batch[index]]
        # print(token_type_ids)
        # print(query_labels)
        # labels = [torch.tensor(example.labels).long() for example in self.query_x_batch[index]]
        # print(labels)
        # print(query_pseudo)
        # pseudo = [torch.tensor(example.pseudo).long() for example in self.query_x_batch[index]]
        # print(pseudo)
        # exit()
        # data = {'token_ids': self.token_ids[index],
        #         'attention_masks': self.attention_masks[index],
        #         'token_type_ids': self.token_type_ids[index]}
        support_data = {'token_ids': support_token_ids,
        'attention_masks': support_attention_masks,
        'token_type_ids': support_token_type_ids,
        'labels':support_labels,
        'pseudo':support_pseudo}
        if self.mode != "train":
            query_data = {'token_ids': query_token_ids,
            'attention_masks': query_attention_masks,
            'token_type_ids': query_token_type_ids,
            'pseudo':query_pseudo}
        else:        
            query_data = {'token_ids': query_token_ids,
            'attention_masks': query_attention_masks,
            'token_type_ids': query_token_type_ids,
            'labels':query_labels,
            'pseudo':query_pseudo}      
        # print(support_data)
        # print(query_data)
        # print(len(support_data['token_ids']))
        # print(len(query_data['token_ids']))
        # exit()
        # if self.ent_type is not None:
        #     data['ent_type'] = self.ent_type[index]

        # if self.labels is not None:
        #     data['labels'] = self.labels[index]

        # if self.pseudo is not None:
        #     data['pseudo'] = self.pseudo[index]

        # if self.start_ids is not None:
        #     data['start_ids'] = self.start_ids[index]
        #     data['end_ids'] = self.end_ids[index]
        query_callback_info = np.array(self.query_x_callback_info_batch[index])
        # print(len(query_data))
        # print(len(query_callback_info))
        # print(type(query_callback_info[0]))
        # print(query_callback_info)
        # exit()
        return support_data, query_data

class NERDataset(Dataset):
    def __init__(self, task_type, features, mode, **kwargs):

        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = None
        self.start_ids, self.end_ids = None, None
        self.ent_type = None
        self.pseudo = None
        if mode == 'train':
            self.pseudo = [torch.tensor(example.pseudo).long() for example in features]
            if task_type == 'crf':
                self.labels = [torch.tensor(example.labels) for example in features]
            else:
                self.start_ids = [torch.tensor(example.start_ids).long() for example in features]
                self.end_ids = [torch.tensor(example.end_ids).long() for example in features]

        if kwargs.pop('use_type_embed', False):
            self.ent_type = [torch.tensor(example.ent_type) for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.ent_type is not None:
            data['ent_type'] = self.ent_type[index]

        if self.labels is not None:
            data['labels'] = self.labels[index]

        if self.pseudo is not None:
            data['pseudo'] = self.pseudo[index]

        if self.start_ids is not None:
            data['start_ids'] = self.start_ids[index]
            data['end_ids'] = self.end_ids[index]

        return data



