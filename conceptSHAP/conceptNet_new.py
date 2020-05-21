import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, combinations
# DEBUG
import IPython
import numpy as np
import math
e = IPython.embed

class ConceptNet_New(nn.Module):

    def __init__(self, n_concepts, train_embeddings):

        super(ConceptNet_New, self).__init__()
        embedding_dim = train_embeddings.shape[1]
        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)
        self.n_concepts = n_concepts
        self.train_embeddings = train_embeddings.transpose(0, 1) # (dim, all_data_size)

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept

    def forward(self, train_embedding, h_x):
        """
        train_embedding: shape (bs, embedding_dim)
        """
        # calculating projection of train_embedding onto the concept vector space
        proj_matrix = (self.concept @ torch.inverse((self.concept.T @ self.concept))) \
                      @ self.concept.T # (embedding_dim x embedding_dim)
        proj = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)

        # passing projected activations through rest of model
        y_pred = h_x(proj.T)
        orig_pred = h_x(train_embedding)

        ###### Calculate the regularization terms in second version of paper
        # new parameters
        k = 10 # TODO this param is NOT tuned yet, but it might be very important

        ### calculate first regularization term, to be maximized
        # 1. find the top k nearest neighbour
        all_concept_knns = []
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)

            # cosine dist
            # dot_prods = torch.sum(c * self.train_embeddings, dim=0)
            # denoms = torch.norm(c) * torch.norm(self.train_embeddings, dim=0)
            # distance = dot_prods / denoms

            # euc dist
            distance = torch.norm(self.train_embeddings - c, dim=0) # (num_total_activations)

            knn = distance.topk(k, largest=False)
            indices = knn.indices # (k)
            knn_activations = self.train_embeddings[:, indices] # (activation_dim, k)
            all_concept_knns.append(knn_activations)

        # 2. calculate the avg dot product for each concept with each of its knn
        L_sparse_1_new = 0.0
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)
            c_knn = all_concept_knns[concept_idx] # knn for c
            dot_prod = torch.sum(c * c_knn) / k # avg dot product on knn
            L_sparse_1_new += dot_prod
        L_sparse_1_new = L_sparse_1_new / self.n_concepts

        ### calculate Second regularization term, to be minimized
        all_concept_dot = self.concept.T @ self.concept
        # all_concept_dot = torch.abs(all_concept_dot)
        mask = torch.eye(self.n_concepts).cuda() * -1 + 1 # mask the i==j positions
        L_sparse_2_new = torch.mean(all_concept_dot * mask)

        norm_metrics = torch.mean(all_concept_dot * torch.eye(self.n_concepts).cuda())

        return orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, [norm_metrics]

    def loss(self, train_embedding, train_y_true, h_x, regularize=False, doConceptSHAP=False, l_1=5., l_2=5.):
        """
        This function will be called externally to feed data and get the loss
        """
        # TODO hardcoded param: result is extremely sensitive to them
        l_1 = 1/1000
        l_2 = 1/500 # it is important to MAKE SURE L2 GOES DOWN! that will let concepts separate from each other

        orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, metrics = self.forward(train_embedding, h_x)

        ce_loss = nn.CrossEntropyLoss()
        loss_new = ce_loss(y_pred, train_y_true)
        pred_loss = torch.mean(loss_new)

        # completeness score
        def n(y_pred):
            orig_correct = torch.sum(train_y_true == torch.argmax(orig_pred, axis=1))
            new_correct = torch.sum(train_y_true == torch.argmax(y_pred, axis=1))
            return torch.div(new_correct, orig_correct)

        completeness = n(y_pred)

        conceptSHAP = []
        if doConceptSHAP:
            def proj(concept):
                proj_matrix = (concept @ torch.inverse((concept.T @ concept))) \
                              @ concept.T  # (embedding_dim x embedding_dim)
                proj = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)

                # passing projected activations through rest of model
                return h_x(proj.T)

            # shapley score
            c_id = np.asarray(list(range(len(self.concept.T))))
            for idx in c_id:
                exclude = np.delete(c_id, idx)
                subsets = np.asarray(self.powerset(list(exclude)))
                sum = 0
                for subset in subsets:
                    # score 1:
                    c1 = subset + [idx]
                    concept = np.take(self.concept.T.detach().cpu().numpy(), np.asarray(c1), axis=0)
                    concept = torch.from_numpy(concept).T
                    pred = proj(concept.cuda())
                    score1 = n(pred)

                    # score 2:
                    c1 = subset
                    concept = np.take(self.concept.T.detach().cpu().numpy(), np.asarray(c1), axis=0)
                    concept = torch.from_numpy(concept).T
                    pred = proj(concept.cuda())
                    score2 = n(pred)

                    norm = (math.factorial(len(c_id) - len(subset) - 1) * math.factorial(len(subset))) / \
                           math.factorial(len(c_id))
                    sum +=  norm * (score1.data.item() - score2.data.item())
                conceptSHAP.append(sum)

        if regularize:
            final_loss = pred_loss + (l_1 * L_sparse_1_new * -1) + (l_2 * L_sparse_2_new)
        else:
            final_loss = pred_loss

        return completeness, conceptSHAP, final_loss, pred_loss, L_sparse_1_new, L_sparse_2_new, metrics

    def powerset(self, iterable):
        "powerset([1,2,3]) --> [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]"
        s = list(iterable)
        pset = chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
        return [list(i) for i in list(pset)]