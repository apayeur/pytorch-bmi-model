import torch.nn as nn
import torch
import torch.nn.functional as F
from effectors import NoEffector

"""
Description:
-----------
All networks described below follow the following design principles, except NoisyRNN.
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


class NoisyRNN(nn.RNN):
    r"""
    Noisy RNN with decay. To replace native `nn.RNN` from Pytorch.

    :math:`v_t = v_{t-1} + \\alpha (-v_{t-1} + Wf(v_{t-1}) + U x_t + b_h + b_i) + \sigma \sqrt{\\alpha} \xi_t`

    where :math:`\xi_t` = standard Gaussian variate
    """
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', 0.2)
        self.sigma = kwargs.pop('sigma', 5e-3)
        super().__init__(*args, **kwargs)
        assert self.num_layers == 1, "No support for multiple layers"
        assert not self.bidirectional, "No support for bidirectionality for NoisyRNN"
        assert self.dropout == 0, "No support for dropout"

    def forward(self, x, v_prev=None):
        assert (x.dim() in (2, 3)), f"RNN: Expected input to be 2-D or 3-D but received {x.dim()}-D tensor"
        assert self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU'

        if x.dim() == 3:
            if v_prev is None:
                v_prev = torch.zeros(1, self.hidden_size, dtype=x.dtype, device=x.device)
            output = torch.empty((x.shape[0], x.shape[1], self.hidden_size))

            if self.mode == 'RNN_TANH':
                v = v_prev
                for t in range(x.size(1)):
                    xi = torch.randn((1, self.hidden_size))
                    v = v + self.alpha * (-v + F.tanh(v) @ self.weight_hh_l0.T +
                                              x[:, t, :] @ self.weight_ih_l0.T +
                                              self.bias_ih_l0 + self.bias_hh_l0) + self.sigma*self.alpha**0.5*xi
                    output[:, t, :] = F.tanh(v)
            elif self.mode == 'RNN_RELU':
                v = v_prev
                for t in range(x.size(1)):
                    xi = torch.randn((x.shape[0], self.hidden_size))
                    v = v + self.alpha * (-v + F.relu(v) @ self.weight_hh_l0.T +
                                          x[:, t, :] @ self.weight_ih_l0.T +
                                          self.bias_ih_l0 + self.bias_hh_l0) + self.sigma * self.alpha ** 0.5 * xi
                    output[:, t, :] = F.relu(v)
            return output, output[:, -1, :] if self.batch_first else output[-1, :, :]

        elif x.dim() == 2:
            # x has shape (N, `self.input_size)
            if v_prev is None:
                v_prev = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
            output = torch.empty((x.shape[0], self.hidden_size))
            if self.mode == 'RNN_TANH':
                v = v_prev
                xi = torch.randn((x.shape[0], self.hidden_size))
                v = v + self.alpha * (-v + F.tanh(v) @ self.weight_hh_l0.T +
                                          x @ self.weight_ih_l0.T +
                                          self.bias_ih_l0 + self.bias_hh_l0) + self.sigma*self.alpha**0.5*xi
                output = v  # not that v is return when input is 2D
            elif self.mode == 'RNN_RELU':
                v = v_prev
                xi = torch.randn((1, self.hidden_size))
                v = v + self.alpha * (-v + F.relu(v) @ self.weight_hh_l0.T +
                                      x @ self.weight_ih_l0.T +
                                      self.bias_ih_l0 + self.bias_hh_l0) + self.sigma * self.alpha ** 0.5 * xi
                output = v
            return output, None


class SimpleNet(nn.Module):
    """
    Simplest network. The `motor_cortex` is an instance of the `nn.RNN` class from pytorch.
    """
    def __init__(self, network_size=(5, 100, 200, 2), nonlinearity='relu'):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.network_size = network_size
        self.input_encoder = nn.Linear(network_size[0], network_size[1])
        self.input_activation = nn.ReLU() if nonlinearity == 'relu' else nn.Tanh()
        self.motor_cortex = nn.RNN(input_size=network_size[1], hidden_size=network_size[2],
                                   nonlinearity=nonlinearity, batch_first=True)
        self.readout_layer = nn.Linear(network_size[2], network_size[3], bias=False)
        self.effector = NoEffector(workspace_dim=network_size[-1])  # placeholder for more sophisticated effectors, like an arm or a BCI decoder

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        # -- Weights --
        # Encoder
        nn.init.uniform_(self.input_encoder.weight, a=0, b=(24. / self.input_encoder.weight.size(1)) ** 0.5)
                        #std=(2. / self.input_encoder.weight.size(1)) ** 0.5)
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

    def forward(self, x, h0):
        x = self.input_encoder(x)
        x = self.input_activation(x)
        h, _ = self.motor_cortex(x, h0)
        controls = self.readout_layer(h)
        x = self.effector(controls)
        return x, h, controls


class NoisyNet(SimpleNet):
    def __init__(self, network_size=(5, 100, 200, 2), nonlinearity='relu'):
        super().__init__(network_size, nonlinearity)
        self.motor_cortex = NoisyRNN(input_size=network_size[1], hidden_size=network_size[2],
                                     nonlinearity='relu', batch_first=True)


class NoisyNetWithFeedback(NoisyNet):
    def __init__(self, network_size=(6, 100, 200, 2), nonlinearity='relu', delay=0, feedback_type='position_only', distance=0.07):
        super().__init__(network_size, nonlinearity)
        self.delay = delay  # in time steps
        self.feedback_type = feedback_type
        self.distance = distance  # in meters, needed for proper normalization of position feedback
        assert feedback_type in ['position_only', 'position_and_velocity']

    def forward(self, x, v0):
        assert x.dim() == 3, f"RNN: Expected input to be 3-D but received {x.dim()}-D tensor"
        assert v0.dim() == 3, f"RNN: Expected second input to be 3-D but received {v0.dim()}-D tensor"
        assert self.nonlinearity == 'tanh' or self.nonlinearity == 'relu'

        if v0 is None:
            v0 = torch.zeros(x.dim(0), self.network_size[2], dtype=x.dtype, device=x.device)
        else:
            v0 = v0.squeeze()
        # Output: hidden-layer activity, effector states and controls
        h = torch.empty((x.shape[0], x.shape[1], self.network_size[2]))
        effector_states = torch.empty((x.shape[0], x.shape[1], 3*self.effector.workspace_dim))
        controls = torch.empty((x.shape[0], x.shape[1], self.effector.workspace_dim))

        # Generate the random initial effector state
        fake_controls = torch.zeros((x.size(0), 1, self.effector.workspace_dim))
        effector_initial_state = self.effector(fake_controls)
        effector_initial_state = effector_initial_state[:, 0, :]

        # Construct buffer for delay feedback
        if self.feedback_type == 'position_only':
            feedback_buffer = [effector_initial_state[:, :self.effector.workspace_dim] / self.distance] * (int(self.delay) + 1)
        elif self.feedback_type == 'position_and_velocity':
            feedback_buffer = [torch.cat((effector_initial_state[:, :self.effector.workspace_dim] / self.distance,
                                          effector_initial_state[:, self.effector.workspace_dim:2*self.effector.workspace_dim]), -1)] * (int(self.delay) + 1)

        # Integrate
        v = v0
        s = effector_initial_state
        for t in range(x.size(1)):
            x_cat = torch.cat((x[:, t, :], feedback_buffer.pop(0)), -1)
            x_cat = self.input_encoder(x_cat)
            x_cat = self.input_activation(x_cat)
            v, _ = self.motor_cortex(x_cat, v)
            if self.nonlinearity == 'tanh':
                v = F.tanh(v)
            elif self.nonlinearity == 'relu':
                v = F.relu(v)
            c = self.readout_layer(v)
            s = self.effector(c, s)

            effector_states[:, t, :] = s
            controls[:, t, :] = c
            h[:, t, :] = v

            if self.feedback_type == 'position_only':
                feedback_buffer.append(s[:, :self.effector.workspace_dim] / self.distance)
            elif self.feedback_type == 'position_and_velocity':
                feedback_buffer.append(torch.cat((s[:, :self.effector.workspace_dim] / self.distance,
                                                  s[:, self.effector.workspace_dim:2*self.effector.workspace_dim]), -1))

        return effector_states, h, controls






