from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
from copy import deepcopy
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import AdamW
import time
class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, model):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.outer_update_lr  = args.meta_lr
        self.inner_update_lr  = args.update_lr
        self.inner_update_step = args.update_step
        self.inner_update_step_eval = args.update_step_test
        self.device = int(args.gpu_ids)
        self.task_num = args.task_num
        self.T = args.T
        self.alpha = args.alpha


        self.model = model
        # self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr)
        self.outer_optimizer = AdamW(self.model.parameters(), lr=self.outer_update_lr, eps=args.adam_epsilon)
        self.model.train()

    def forward(self, x_spt, x_qry, training = True):
        """
        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]
        
        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        sum_gradients = []
        all_outputs = []
        num_task = self.task_num
        num_inner_update_step = self.inner_update_step if training else self.inner_update_step_eval
        kl_criterion = nn.KLDivLoss()
        # for task_id, task in enumerate(self.task_num):
        if not training:
            self.task_num = 5
        for task_id in range(self.task_num):
            support = {key: x_spt[key][task_id] for key in x_spt}
            query   = {key: x_qry[key][task_id] for key in x_qry}
            
            fast_model = deepcopy(self.model)
            fast_model.to(self.device)
            # support_dataloader = DataLoader(support, sampler=RandomSampler(support),
            #                                 batch_size=self.inner_batch_size)
            
            # inner_optimizer = Adam(fast_model.parameters(), lr=self.inner_update_lr)
            inner_optimizer = AdamW(fast_model.parameters(), lr=self.inner_update_lr,eps=self.args.adam_epsilon)
            fast_model.train()
            
            print('----Task',task_id, '----')
            if not training:
                self.model.eval()
                teacher_outputs = self.model(**query)
                teacher_scores = teacher_outputs[1]
                # print(teacher_scores.shape)
                # exit()
                # self.model.train()
            for i in range(0,num_inner_update_step):
                # all_loss = []
                # for inner_step, batch in enumerate(support_dataloader):
                outputs = fast_model(**support)

                loss = outputs[0]

                if not training:
                    # print(outputs[1].shape)
                    # exit()
                    student_outputs = fast_model(**query)
                    p = F.log_softmax(student_outputs[1]/self.T,dim =2)
                    q = F.softmax(teacher_scores/self.T,dim =2)
                    # print(p.shape)
                    # print(q.shape)
                    kl_loss = kl_criterion(p,q)*self.T*self.T
                    # loss += kl_loss
                    loss = (1-self.alpha)*loss+self.alpha*kl_loss
                    loss.backward(retain_graph = True)
                    # loss = (1-alpha)*loss
                    # kl_loss = alpha*kl_loss
                    # loss.backward(retain_graph = True)
                    # kl_loss.backward()
                else:
                    loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()
                
                if (i+1) % 4 == 0:
                    print("Inner Loss: ", loss)

            q_outputs = fast_model(**query)

            if training:
                q_loss = q_outputs[0]
                q_loss.backward()
                fast_model.to(torch.device('cpu'))

                for i, params in enumerate(fast_model.parameters()):
                    if task_id == 0:
                        sum_gradients.append(deepcopy(params.grad))
                    else:
                        sum_gradients[i] += deepcopy(params.grad)
            else:
                all_outputs.extend(q_outputs[0])
            # q_logits = F.softmax(q_outputs[1],dim=1)
            # pre_label_id = torch.argmax(q_logits,dim=1)
            # pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            # q_label_id = q_label_id.detach().cpu().numpy().tolist()
            
            # acc = accuracy_score(pre_label_id,q_label_id)
            # task_accs.append(acc)
            
            del fast_model, inner_optimizer
            # torch.cuda.empty_cache()
            # print("sleep")
            # time.sleep(5000)


        if training:
            # Average gradient across tasks
            for i in range(0,len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            #Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            
            del sum_gradients
            gc.collect()
        # print("sleep")
        # time.sleep(5000)
        if training:
            return q_outputs[0]
        else:
            # return q_outputs
            return all_outputs
        # return np.mean(task_accs)