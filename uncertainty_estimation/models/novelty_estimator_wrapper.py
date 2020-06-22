class NoveltyEstimator:
    def __init__(self, model_type, model_params, train_params, method_name):
        self.model_type = model_type
        self.name = method_name
        self.model_params = model_params
        self.train_params = train_params

    def train(self, train_data):
        if self.name == 'AE':
            self.model = self.model_type(**self.model_params, train_data=train_data)
            self.model.train(**self.train_params)
        elif self.name == 'sklearn':
            self.model = self.model_type(**self.model_params)

            self.model.fit(train_data)

    def get_novelty_score(self, data):
        if self.name == 'AE':
            return self.model.get_reconstr_error(data)
        elif self.name == 'sklearn':
            return - self.model.score_samples(data)
