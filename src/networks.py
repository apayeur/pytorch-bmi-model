import torch.nn as nn

"""
Description:
-----------
Pretty much all networks described below follow the following design principles.
1) Each network contains the following modules
    - input_encoder :   Takes workspace and task variables (target position, context, effector position, hold signal) 
                        as input and output a higher-dimensional embedding of these variables. 
    - motor_cortex :    The recurrent network. Takes the signal encoded by the `input_encoder` as input. 
    - readout_layer :   Linear readout of the recurrent network's hidden layer activity. Produces the controls that 
                        activate the chosen effector.
    - effector :        Either an arm model (for manual task) or a BCI decoder (bci task) or an Identity placeholder 
                        when the output of the `readout_layer` is directly used as the effector.
                        
2) Some parameters are frozen (not learned) by design.
    Not learned : parameters of the `input_encoder`, of the `readout_layer` and of the `effector`. 
"""


class SimpleNet(nn.Module):
    def __init__(self, network_size=(5, 100, 200, 2), nonlinearity='relu'):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.network_size = network_size
        self.input_encoder = nn.Linear(network_size[0], network_size[1])
        self.input_activation = nn.ReLU() if nonlinearity=='relu' else nn.Tanh()
        self.motor_cortex = nn.RNN(input_size=network_size[1], hidden_size=network_size[2],
                                   nonlinearity=nonlinearity, batch_first=True)
        self.readout_layer = nn.Linear(network_size[2], network_size[3], bias=False)
        self.effector = nn.Identity()  # placeholder for more sophisticated effectors, like an arm or a BCI decoder

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        # -- Weights --
        # Encoder
        nn.init.normal_(self.input_encoder.weight,
                        std=(2. / self.input_encoder.weight.size(1)) ** 0.5)
        # Motor cortex
        nn.init.uniform_(self.motor_cortex.weight_ih_l0,
                         a=-1. / self.motor_cortex.weight_ih_l0.size(1) ** 0.5,
                         b=1. / self.motor_cortex.weight_ih_l0.size(1) ** 0.5)  # input weight matrix
        nn.init.normal_(self.motor_cortex.weight_hh_l0,
                        std=1. / self.motor_cortex.weight_hh_l0.size(1) ** 0.5)  # recurrent weight matrix
        # Readout layer
        nn.init.normal_(self.readout_layer.weight, std=0.1 / self.readout_layer.weight.size(1))

        # -- Biases --
        if self.nonlinearity == 'relu':
            # Encoder
            nn.init.uniform_(self.input_encoder.bias, a=0.5, b=1.5)
            # Motor cortex
            nn.init.zeros_(self.motor_cortex.bias_ih_l0)  # bias from input layer to hidden
            nn.init.uniform_(self.motor_cortex.bias_hh_l0, a=0.5, b=1.5)  # bias from hidden layer to hidden
        elif self.nonlinearity == 'tanh':
            # Encoder
            nn.init.zeros_(self.input_encoder.bias)
            # Motor cortex
            nn.init.zeros_(self.motor_cortex.bias_ih_l0)  # bias from input layer to hidden
            nn.init.zeros_(self.motor_cortex.bias_hh_l0)  # bias from hidden layer to hidden

    def _freeze_params(self):
        self.input_encoder.weight.requires_grad = False
        self.input_encoder.bias.requires_grad = False
        self.readout_layer.weight.requires_grad = False
        self.motor_cortex.bias_ih_l0.requires_grad = False

    def forward(self, x, h0, effector_state=None):
        x = self.input_encoder(x)
        x = self.input_activation(x)
        h, _ = self.motor_cortex(x, h0)
        x = self.readout_layer(h)
        if effector_state is None:
            x = self.effector(x)
        else:
            x = self.effector(x, effector_state)
        return x, h


