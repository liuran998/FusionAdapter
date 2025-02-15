from embedding import *
from collections import OrderedDict
import torch


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))


        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)


        # self.image = nn.Embedding(6555, 4096)
        # self.text = nn.Embedding(6555, 300)
        #
        # self.text2emb = dataset['text2emb']
        # self.visual2emb = dataset['visual2emb']
        # self.image.weight.data.copy_(torch.from_numpy(self.visual2emb))
        # self.text.weight.data.copy_(torch.from_numpy(self.text2emb))

        self.ling_align = nn.Linear(300, 100)
        self.visual_align = nn.Linear(4096, 100)

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'umls-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'WN9-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'FB-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)

        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.rel_s_sharing = dict()

        # # Use contrastive loss
        # def contrastive_loss(modality, anchor):
        #     positive_similarity = torch.nn.functional.cosine_similarity(modality, anchor, dim=-1)
        #     negative_samples = torch.cat([anchor.roll(i, 0) for i in range(1, 5)], dim=0)
        #     negative_similarity = torch.nn.functional.cosine_similarity(modality.unsqueeze(1), negative_samples,
        #                                                                 dim=-1)
        #     return (1 - positive_similarity.mean()) + negative_similarity.mean()


    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task,adaptor_image, adaptor_text, iseval=False, curr_rel=''):

        support, support_negative, query, negative = [
            self.embedding(t, adaptor_image, adaptor_text) for t in task
        ]

        # support, support_negative, query, negative = [self.embedding(t,adaptor_image, adapter_text) for t in task]

        # Combined embedding
        support_combined = support[0]
        support_negative_combined = support_negative[0]
        # Structural embeddings
        support_structural = support[1]
        support_negative_structural = support_negative[1]
        # Image embeddings
        support_image = support[2]
        support_negative_image = support_negative[2]
        # Text embeddings
        support_text = support[3]
        support_negative_text = support_negative[3]

        query_combined = query[0]
        negative_combined = negative[0]

        few = support_combined.shape[1]  # num of few
        num_sn = support_negative_combined.shape[1]  # num of support negative
        num_q = query_combined.shape[1]  # num of query
        num_n = negative_combined.shape[1]  # num of query negative

        rel = self.relation_learner(support_combined)
        # rel = self.delta*adaptor(rel) + (1-self.delta)*rel
        rel.retain_grad()

        # weights = self.relation_learner.rel_fc1.fc

        # relation for support
        rel_s = rel.expand(-1, few + num_sn, -1, -1)

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support_combined, support_negative_combined)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)

                y = torch.Tensor([1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta * grad_meta
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query_combined, negative_combined)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        return p_score, n_score, support_structural, support_negative_structural, support_image, support_negative_image, \
               support_text, support_negative_text

    def task_adaptor(self, task, adaptor_image,adaptor_text,iseval=False, curr_rel=''):

        support, support_negative, _, _ = [
            self.embedding(t, adaptor_image, adaptor_text) for t in task
        ]

        support_combined = support[0]
        support_negative_combined = support_negative[0]

        few = support_combined.shape[1]              # num of few
        num_sn = support_negative_combined.shape[1]  # num of support negative
        # num_v = valid.shape[1]              # num of query
        # num_v_neg = valid_negative.shape[1]           # num of query negative

        rel = self.relation_learner(support_combined)
        # rel = self.delta*adaptor(rel) + (1-self.delta)*rel
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_s_sharing.keys():
            rel_s = self.rel_s_sharing[curr_rel]
        else:
            if not self.abla:
                sup_neg_e1, sup_neg_e2 = self.split_concat(support_combined, support_negative_combined)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)
                y = torch.Tensor([1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_s = rel - self.beta * grad_meta
            else:
                rel_s = rel

            self.rel_s_sharing[curr_rel] = rel_s

        rel_s = rel_s.expand(-1, few+num_sn, -1, -1)

        sup_neg_e1, sup_neg_e2 = self.split_concat(support_combined, support_negative_combined)
        p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)


        return p_score, n_score
