from threading import enumerate
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    copy import deepcopy



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, net, optimizer):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        # self.n_way = args.n_way
        # self.k_spt = args.k_spt
        # self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = net
        self.meta_optim = optimizer
        # self.scheduler = scheduler

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def parameter_replace(self,replace_weight):
        num=0
        for key in self.net.state_dict().keys():
            if key == "bert_module.embeddings.position_ids":
                continue
            # self.net.state_dict()[key].copy_(replace_weight[num])
            self.net.state_dict()[key]=replace_weight[num]
            num+=1        

    def forward(self, x_spt, x_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        # task_num, setsz, c_, h, w = x_spt.size()
        # querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        orignal_weights = list(self.net.parameters())
        # print(task_num,len(x_spt))
        # print(x_spt[0],x_spt[0].shape) #5,3,84,84
        # exit()
        for i in range(self.task_num):

            # 1. run the i-th task and compute loss for k=0
            # logits = self.net(x_spt[i], vars=None, bn_training=True)
            # # print(logits.shape) #5,5
            # # exit()
            # # print(y_spt[i].shape) #5
            # # exit()
            # loss = F.cross_entropy(logits, y_spt[i])
            x_spt_i = {key: x_spt[key][i] for key in x_spt}
            x_qry_i = {key: x_qry[key][i] for key in x_qry}
            # print(self.net)
            loss = self.net(**x_spt_i)[0]
            grad = torch.autograd.grad(loss, self.net.parameters(),allow_unused=True)
            # print(grad)
            # print(self.net.state_dict())
            # exit()
            # print(type(grad),len(grad))
            # print(len(self.net.state_dict()))
            # for key in self.net.state_dict().keys():
            #     print(key,self.net.state_dict()[key].shape)
            # num=0
            # for key in self.net.state_dict().keys():
            #     if key == "bert_module.embeddings.position_ids":
            #         continue
            #     print(num,key)
            #     num+=1
            # for item in self.net.parameters():
            #     print(item.shape)
            # exit()
            # grad = torch.autograd.grad(loss, self.net.parameters())
            # print(grad)
            # 输出grad数据和网络模型尺寸
            # print(self.net.parameters()[0])
            print("grad_Ef==================")
            print(self.net(**x_spt_i)[0])
            exit()
            # for p in zip(grad, self.net.parameters()):
            #     print(p[1])
            #     print(p[0])
            #     if p[0]==None:
            #         print("None",p[1].size())
            #         print(type(p[1]))
            #     else:
            #         print(p[0].size(),p[1].size())
            # exit()
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            # print(fast_weights[0])
            # exit()
            # print(len(list(self.net.parameters())))
            # print(type(fast_weights))
            # print(len(fast_weights))
            # exit()

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                # logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                # loss_q = F.cross_entropy(logits_q, y_qry[i])
                loss_q = self.net(**x_qry_i)[0]
                losses_q[0] += loss_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                # self.net.parameters() = fast_weights
                self.parameter_replace(fast_weights)
                # num=0
                # for key in self.net.state_dict().keys():
                #     if key == "bert_module.embeddings.position_ids":
                #         continue
                #     self.net.state_dict()[key].copy_(fast_weights[num])
                #     num+=1
                loss_q = self.net(**x_qry_i)[0]
                # loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
            
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                loss = self.net(**x_spt_i)[0]
                # loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, self.net.parameters(),allow_unused=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                self.parameter_replace(fast_weights)
                # num=0
                # for key in self.net.state_dict().keys():
                #     if key == "bert_module.embeddings.position_ids":
                #         continue
                #     self.net.state_dict()[key].copy_(fast_weights[num])
                #     num+=1
                loss_q = self.net(**x_qry_i)[0]
                # loss_q will be overwritten and just keep the loss_q on last update step.
                # loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                # with torch.no_grad():
                #     pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                #     correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                #     corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / self.task_num
        self.parameter_replace(orignal_weights)
        # num=0
        # for key in self.net.state_dict().keys():
        #     if key == "bert_module.embeddings.position_ids":
        #         continue
        #     self.net.state_dict()[key].copy_(orignal_weights[num])
        #     num+=1
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        # self.scheduler.step()


        # accs = np.array(corrects) / (querysz * task_num)

        return loss_q


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
