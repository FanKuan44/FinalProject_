def get_acc_predictor(model, inputs, targets, verbose=False):
    if model == 'mlp':
        from acc_predictor.mlp import MLP
        acc_predictor = MLP(n_feature=inputs.shape[1])
        acc_predictor.fit(x=inputs, y=targets, verbose=verbose)
    else:
        raise NotImplementedError

    return acc_predictor

