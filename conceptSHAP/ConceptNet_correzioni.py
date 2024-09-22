import torch
import torch.nn as nn
from itertools import chain, combinations
import numpy as np
import math
import torch.nn.functional as F
import itertools

class ConceptNet(nn.Module):
    def __init__(self, n_concepts, train_embeddings, num_classes, bge_model, original_texts):
        super(ConceptNet, self).__init__()
        embedding_dim = train_embeddings.shape[1]
        self.concept = nn.Parameter(torch.randn(embedding_dim, n_concepts))
        self.n_concepts = n_concepts
        self.train_embeddings = train_embeddings.T
        self.num_classes = num_classes
        self.bge_model = bge_model
        self.original_texts = original_texts
        self.linear = nn.Linear(embedding_dim, num_classes)

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept
    def get_top_words_for_concept(self, concept_idx, train_embedding, topk):
        concept = self.concept[:, concept_idx]
        
        # Controlla se train_embedding è un singolo embedding o un batch
        if train_embedding.dim() == 1:
            train_embedding = train_embedding.unsqueeze(0)
        
        similarities = torch.mm(train_embedding, concept.unsqueeze(1)).squeeze()
        
        # Se abbiamo un batch, prendiamo la media delle similarità
        if similarities.dim() > 1:
            similarities = similarities.mean(dim=0)
        
        top_indices = torch.topk(similarities, min(topk, len(similarities))).indices
        
        # Usa i testi originali invece degli embedding
        top_texts = [self.original_texts[idx] for idx in top_indices]
        
        # Tokenizza i testi più simili
        tokenizer = self.bge_model.tokenizer
        all_tokens = [token for text in top_texts for token in tokenizer.tokenize(text)]
        
        # Prendi le prime topk parole uniche
        unique_tokens = list(dict.fromkeys(all_tokens))
        return unique_tokens[:topk]

    def forward(self, train_embedding, h_x, topk):
        print(f"train_embedding shape: {train_embedding.shape}")
        print(f"self.train_embeddings shape: {self.train_embeddings.shape}")
        print(f"self.concept shape: {self.concept.shape}")
        # calculating projection of train_embedding onto the concept vector space
        proj_matrix = (self.concept @ torch.inverse((self.concept.T @ self.concept))) \
                      @ self.concept.T # (embedding_dim x embedding_dim)
        proj = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)
    
        # passing projected activations through rest of model
        y_pred = self.linear(proj.T)
    
        orig_pred = self.linear(train_embedding)
    
        # Calculate the regularization terms as in new version of paper
        k = topk # this is a tunable parameter
    
        ### calculate first regularization term, to be maximized
        # 1. find the top k nearest neighbour
        all_concept_knns = []
        for concept_idx in range(self.n_concepts):
            c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)
    
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
        mask = torch.eye(self.n_concepts).cuda() * -1 + 1 # mask the i==j positions
        L_sparse_2_new = torch.mean(all_concept_dot * mask)

        norm_metrics = torch.mean(all_concept_dot * torch.eye(self.n_concepts).cuda())
        similarity_penality = torch.mean(torch.abs(torch.matmul(self.concept.T, self.concept) - torch.eye(self.n_concepts).to(self.concept.device)))

        metrics = [norm_metrics, similarity_penality]
        
        return orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, metrics

    
    def calculate_concept_diversity(self):
        all_words = set()
        concept_words = []
    
        for i in range(self.n_concepts):
            top_words = self.get_top_words_for_concept(i, self.train_embeddings.T, 25)
            concept_words.append([(word, 1) for word in top_words])  # Simuliamo il conteggio
            all_words.update(top_words)
        
        unique_words = 0
        for concept in concept_words:
            concept_words_set = set([word for word, _ in concept])
            unique_words += len(concept_words_set - (all_words - concept_words_set))
        
        diversity = unique_words / self.n_concepts
        return diversity
    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
        
    def proj_and_predict(self, concept, train_embedding):
        proj_matrix = (concept @ torch.inverse((concept.T @ concept))) @ concept.T
        proj = proj_matrix @ train_embedding.T
        y_pred = self.linear(proj.T)
        return y_pred
    
    def loss(self, train_embedding, train_y_true, h_x, regularize, doConceptSHAP, l_1, l_2, topk):
        orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new, metrics = self.forward(train_embedding, h_x, topk)
        
        ce_loss = nn.CrossEntropyLoss()
        loss_new = ce_loss(y_pred, train_y_true)
        pred_loss = torch.mean(loss_new)
    
        # completeness score
        def n(y_pred):
            orig_correct = torch.sum(train_y_true == torch.argmax(orig_pred, axis=1))
            new_correct = torch.sum(train_y_true == torch.argmax(y_pred, axis=1))
            return torch.div(new_correct - (1/self.n_concepts), orig_correct - (1/self.n_concepts))
    
        completeness = n(y_pred)
        conceptSHAP = []
    
        if doConceptSHAP:
            # Approximate SHAP values using sampling
            num_samples = 100  # Numero di campioni da usare per l'approssimazione
            c_id = list(range(len(self.concept.T)))
    
            for idx in c_id:
                shap_value = 0.0
                for _ in range(num_samples):
                    # Genera una coalizione casuale che non include il concetto corrente
                    subset = [i for i in c_id if i != idx and np.random.rand() > 0.5]
                    # Calcola la previsione con e senza il concetto corrente
                    c_with = subset + [idx]
                    c_without = subset
    
                    # Costruisci il concetto con e senza il concetto corrente
                    concept_with = self.concept[:, c_with]
                    concept_without = self.concept[:, c_without] if c_without else None
    
                    # Previsione con il concetto corrente
                    y_pred_with = self.proj_and_predict(concept_with, train_embedding)
                    score_with = n(y_pred_with)
    
                    # Previsione senza il concetto corrente
                    if concept_without is not None:
                        y_pred_without = self.proj_and_predict(concept_without, train_embedding)
                        score_without = n(y_pred_without)
                    else:
                        # Se non ci sono concetti nella coalizione, score_without è 0
                        score_without = torch.tensor(0.0).to(self.concept.device)
    
                    # Aggiorna il valore SHAP per il concetto corrente
                    shap_value += (score_with - score_without).item()
    
                # Calcola il valore medio SHAP
                shap_value /= num_samples
                conceptSHAP.append(shap_value)    
        concept_diversity = self.calculate_concept_diversity()
    
        if regularize:
            diversity_weight = 0.1  # Puoi regolare questo peso
            final_loss = pred_loss + (l_1 * L_sparse_1_new * -1) + (l_2 * L_sparse_2_new) - (diversity_weight * concept_diversity)
        else:
            final_loss = pred_loss
    
        return completeness, conceptSHAP, final_loss, pred_loss, L_sparse_1_new, L_sparse_2_new, metrics, concept_diversity
