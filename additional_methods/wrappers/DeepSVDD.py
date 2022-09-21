from pyod.models.deep_svdd import DeepSVDD
import math

class DeepSVDD_wrapper(DeepSVDD):
    def __init__(self, n_layers=1, shrinkage_factor=0.3, **args):
        
        self.n_layers = n_layers
        self.shrinkage_factor = shrinkage_factor

        try: 
            del args["hidden_neurons"]
        except KeyError:
            pass
        
        self.args = args
    
    def fit(self, X, y=None):
        
        n_features = X.shape[1]
        
        self.hidden_neurons = [math.ceil(n_features * (1-self.shrinkage_factor)**(i+1)) for i in range(self.n_layers)]
        
        super().__init__(hidden_neurons=self.hidden_neurons,  **self.args)
        
        super().fit(X, y)