# models/

Trained weights of the neural Bayes estimator (PyTorch `state_dict`), one per training seed:
`nbe_seed1.pt` (used for the main results), `nbe_seed2.pt`, `nbe_seed3.pt`.

Each network was trained by `python/train.py` on samples simulated from the GEV prior only; no
rainfall data were used. Load with:

    import torch, sys; sys.path.insert(0, "python"); import nbe
    model = nbe.NBE(); model.load_state_dict(torch.load("models/nbe_seed1.pt"))
    quantiles = nbe.predict(model, y)   # y: array of shape (n_datasets, 39)

Valid only for samples of exactly 39 maxima within the range of the training prior.
