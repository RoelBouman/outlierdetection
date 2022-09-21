from pyod.models.anogan import AnoGAN
import math

class AnoGAN_wrapper(AnoGAN):
    def __init__(self, D_n_layers=1, G_n_layers=1, G_shrinkage_factor=0.3, D_shrinkage_factor=0.3, **args):
        
        self.D_n_layers = D_n_layers
        self.G_n_layers = G_n_layers
        self.D_shrinkage_factor = D_shrinkage_factor
        self.G_shrinkage_factor = G_shrinkage_factor
        
        try: 
            del args["G_layers"]
        except KeyError:
            pass
        
        try: 
            del args["D_layers"]
        except KeyError:
            pass
        
        self.args = args
    
    def fit(self, X, y=None):
        

        n_features = X.shape[1]
        
        self.G_encoder_neurons = [math.ceil(n_features * (1-self.G_shrinkage_factor)**(i+1)) for i in range(self.G_n_layers)]
        
        self.G_decoder_neurons = list(reversed(self.G_encoder_neurons))
        
        self.G_layers = self.G_encoder_neurons + self.G_decoder_neurons
        
        self.D_layers = [math.ceil(n_features * (1-self.D_shrinkage_factor)**(i+1)) for i in range(self.D_n_layers)]
        
        super().__init__(G_layers=self.G_layers, D_layers=self.D_layers, **self.args)
        
        super().fit(X, y)