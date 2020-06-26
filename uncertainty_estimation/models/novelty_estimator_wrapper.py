class NoveltyEstimator:
    """Wrapper class for novelty estimation methods

    Parameters
    ----------
    model_type:
        the model to use, e.g. AE, PCA
    model_params: dict
        The parameters used when initializing the model.
    train_params: dict
        The parameters used when fitting the model.
    method_name: str
        Which type of method: 'AE', or 'sklearn' for a sklearn-style novelty detector.
    """

    def __init__(self, model_type, model_params, train_params, method_name):

        self.model_type = model_type
        self.name = method_name
        self.model_params = model_params
        self.train_params = train_params

    def train(self, train_data):
        """Fit the novelty estimator.
        Parameters
        ----------
        train_data: np.ndarray
            The training data to fit the novelty estimator on.
        """
        if self.name == 'AE':
            self.model = self.model_type(**self.model_params, train_data=train_data)
            self.model.train(**self.train_params)
        elif self.name == 'sklearn':
            self.model = self.model_type(**self.model_params)
            self.model.fit(train_data)

    def get_novelty_score(self, data):
        """Apply the novelty estimator to obtain a novelty score for the data.

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get a novelty score

        Returns
        -------
        np.ndarray
            The novelty estimates.

        """
        if self.name == 'AE':
            return self.model.get_reconstr_error(data)
        elif self.name == 'sklearn':
            return - self.model.score_samples(data)
