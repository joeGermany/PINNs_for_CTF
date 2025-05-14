# The implementation for the Lorenz system is based on the example 
# provided in the DeepXDE documentation: 
# https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/lorenz.inverse.html.

# It might seem slightly unfair that we supplying extra information to the physics-informed 
# neural network model, in this CTF benchmarking task, in the form of the differential equation 
# (even if the exact parameters are not supplied, but learned).
# But, we thought that it would only be fair to include PINNs in this comparision due to its
# widespread popularity and use in the scientific computing spheres.

import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt

# For LORENZ system:

## Learnable parameters (with initial guesses)
## We guess, quite , that the parameters are around the canonically used chaotic Lorenz parameters.
sigma = dde.Variable(1.0) # dde.Variable(10.0)
rho = dde.Variable(1.0) # dde.Variable(28.0)
beta = dde.Variable(1.0) # dde.Variable(2.6)

def Lorenz_pde(x, y):
    """
    Lorenz system ODEs with trainable parameters
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x - sigma * (y2 - y1),
        dy2_x - y1 * (rho - y3) + y2,
        dy3_x - y1 * y2 + beta * y3,
    ]

def boundary(_, on_initial):
    return on_initial

class PINN:
    def __init__(self, pair_id, config, train_data, init_data=None, training_timesteps=None, prediction_timesteps=None, delta_t=None):
        self.pair_id = pair_id
        self.config = config
        self.train_data = train_data
        self.init_data = init_data
        self.training_timesteps = training_timesteps
        self.prediction_timesteps = prediction_timesteps
        self.delta_t = delta_t

        # Choice of Dataset; we need to do this because we have to define the differential equation specific to the dataset
        self.dataset_name = config["dataset"]["name"]
        if self.dataset_name == "ODE_Lorenz":
            self.pde = Lorenz_pde
            self.geom = dde.geometry.TimeDomain(0, prediction_timesteps[-1])
            self.ic = [
                dde.icbc.IC(self.geom, lambda X: self.train_data[0][0][0], boundary, component=0),
                dde.icbc.IC(self.geom, lambda X: self.train_data[0][1][0], boundary, component=1),
                dde.icbc.IC(self.geom, lambda X: self.train_data[0][2][0], boundary, component=2),
            ]
        elif self.dataset_name == "PDE_KS":
            raise NotImplementedError("PDE_KS dataset is not implemented yet for the PINNs model.")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.define_problem()

    def define_problem(self):
        # We add training data as boundary points to add them conveniently in the loss function.
        # print(self.training_timesteps[0])
        observe_t = self.training_timesteps[0].reshape(-1, 1)
        # print(observe_t)

        print("Train_data", self.train_data)
        print("first column: ", self.train_data[0][:, 0])
        print("second column: ", self.train_data[0][:, 1])
        observe_y = [
            self.train_data[0][:, 0].reshape(-1, 1),
            self.train_data[0][:, 1].reshape(-1, 1),
            self.train_data[0][:, 2].reshape(-1, 1)
        ]
        print(observe_y, observe_y[0], observe_y[1])
        self.observe_bc = [
            dde.icbc.PointSetBC(observe_t, observe_y[0], component=0),
            dde.icbc.PointSetBC(observe_t, observe_y[1], component=1),
            dde.icbc.PointSetBC(observe_t, observe_y[2], component=2),
        ]

        print(self.observe_bc)

        self.data = dde.data.PDE(
            self.geom,
            self.pde,
            self.ic + self.observe_bc,
            num_domain=self.config["model"]["num_of_domain_points"], # change these??
            num_boundary=self.config["model"]["num_of_boundary_points"], # change these??
            anchors=observe_t,
        )

    def get_model(self):
        net = dde.nn.FNN([1] + [self.config["model"]["num_of_neurons"]] * self.config["model"]["num_of_layers"] + [3], "tanh", "Glorot uniform")
        return dde.Model(self.data, net)

    def train(self):
        model = self.get_model()

        external_trainable_variables = [sigma, rho, beta]
        variable = dde.callbacks.VariableValue(
            external_trainable_variables, period=600, filename="variables.dat"
        )

        # These could also be added as hyperparameters in the config file.
        loss_weights = [1, 1, 1, 1, 1, 1, 10, 10, 10] # 10 for boundary points, which include all the datset.

        # Train with ADAM optimizer
        model.compile(
            "adam", lr=self.config["model"]["learning_rate"], 
            external_trainable_variables=external_trainable_variables,
            loss_weights=loss_weights,
        )

        losshistory, train_state = model.train(
            iterations=self.config["model"]["epochs"], 
            callbacks=[variable],
            display_every=100
        )

        # Fine-tune with L-BFGS-B
        # model.compile("L-BFGS-B", external_trainable_variables=external_trainable_variables)
        # losshistory, train_state = model.train(callbacks=[variable])

        # print("Learned parameters:")
        self.model = model

        dde.saveplot(losshistory, train_state, issave=True, isplot=False)

    def predict(self) -> np.ndarray:
        """
        PINN predictions.

        Returns:
            np.ndarray: Array of predictions.
        """

        # Train the model.
        self.train()
        print(self.prediction_timesteps)
        predictions = self.model.predict(self.prediction_timesteps.reshape(-1, 1))
        print(predictions)

        # predictions = np.zeros((3, len(self.prediction_timesteps)), dtype=np.float32)
        # # Use initial data for iterative predictions
        # init_data = self.init_data.astype(np.float32)
        # for i, t in enumerate(self.prediction_timesteps):
        #     # Predict the next step
        #     prediction = self.model.predict(init_data)
        #     predictions[:, i] = prediction.ravel()

        #     # Update initial data for the next step
        #     init_data = np.concatenate((init_data[:, 1:], prediction), axis=1)

        return predictions