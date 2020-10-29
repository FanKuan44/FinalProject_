def get_acc_predictor(model, inputs, targets):
    if model == 'mlp':
        from acc_predictor.mlp import MLP
        acc_predictor = MLP(n_feature=inputs.shape[1])
        acc_predictor.fit(x=inputs, y=targets)
    else:
        raise NotImplementedError

    return acc_predictor

