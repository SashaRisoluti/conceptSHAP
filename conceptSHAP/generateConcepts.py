import torch
import conceptNet

def train(train_embeddings, clusters, h_x, n_concepts):
    '''
    :param train_embeddings: tensor of sentence embeddings => (# of examples, embedding_dim)
    :param clusters: tensor of embedding clusters generated by k-means -> (# of n_clusters, # of sentences per cluster, embedding_dim)
    :param h_x: final layers of the transformer
    :param n_concepts: number of concepts to generate
    :return: trained conceptModel
    '''

    lr = 0.001
    batch_size = 50
    epochs = 10
    save_interval = 10
    model = conceptNet(clusters, h_x, n_concepts)
    save_dir = 'checkpoints/'
    optimizer = torch.optim.Adam(conceptNet.parameters(), lr=lr)

    for i in range(epochs):

        # generate training batch
        embedding = train_embeddings.narrow(0, i, i + batch_size)
        loss = model(embedding)

        # update gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # model saving
        if (i + 1) % save_interval == 0:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, save_dir /
                       'conceptSHAP_iter_{:d}.pth'.format(i + 1))
