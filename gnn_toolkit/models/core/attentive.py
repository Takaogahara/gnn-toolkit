import torch
from torch_geometric.nn import AttentiveFP
torch.manual_seed(8)


def Attentive(model_params):
    feature_size = model_params["MODEL_FEAT_SIZE"]
    edge_dim = model_params["MODEL_EDGE_DIM"]

    embedding_size = model_params["MODEL_EMBEDDING_SIZE"]
    n_layers = model_params["MODEL_NUM_LAYERS"]
    dropout_rate = model_params["MODEL_DROPOUT_RATE"]
    timesteps = model_params["MODEL_NUM_TIMESTEPS"]

    model = AttentiveFP(feature_size, embedding_size,
                        out_channels=1,
                        edge_dim=edge_dim,
                        num_layers=n_layers,
                        num_timesteps=timesteps,
                        dropout=dropout_rate)
    return model
