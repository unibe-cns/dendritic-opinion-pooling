import math
import torch


class AbstractConvexCell(torch.nn.Module):
    def __init__(self, in_features_per_dendrite, out_features):
        super().__init__()

        self.in_features_per_dendrite = in_features_per_dendrite
        self.out_features = out_features

        self.EE = 0.0  # mV
        self.EI = -85.0  # mV
        self.gL0 = 0.166667  # nS
        self.gLd = torch.ones(self.n_dendrites) * self.gL0  # nS
        self.EL = -70.0  # mV
        self.gc = torch.ones(self.n_dendrites) * 50000.0 * self.gL0  # nS
        self.input_scale = torch.ones(self.in_features)
        self.lambda_e = 1.0

        self._omegaE = [
            torch.nn.Parameter(
                torch.zeros(
                    self.in_features_per_dendrite[d],
                    self.out_features,
                    dtype=torch.double,
                )
            )
            for d in range(self.n_dendrites)
        ]
        self._omegaI = [
            torch.nn.Parameter(
                torch.zeros(
                    self.in_features_per_dendrite[d],
                    self.out_features,
                    dtype=torch.double,
                )
            )
            for d in range(self.n_dendrites)
        ]

        for d in range(self.n_dendrites):
            self.register_parameter(f"omegaE{d}", self._omegaE[d])
            self.register_parameter(f"omegaI{d}", self._omegaI[d])

        self.init_weights()

        self.alpha = 1.0
        theta = self.EL - 0.
        self.f = (
            lambda u: 1.0
            / self.alpha
            * torch.nn.functional.softplus(self.alpha * (u - theta))
        )
        self.f_inv = (
            lambda r: 1.0 / self.alpha * torch.log(torch.exp(self.alpha * r) - 1.0)
            + theta
        )

    @property
    def in_features(self):
        return sum(self.in_features_per_dendrite)

    @property
    def n_dendrites(self):
        return len(self.in_features_per_dendrite)

    @property
    def min_rate(self):
        return self.f(torch.DoubleTensor([self.EI])).item()

    @property
    def max_rate(self):
        return self.f(torch.DoubleTensor([self.EE])).item()

    def assert_valid_rate(self, r):
        assert torch.all(self.min_rate <= r)
        assert torch.all(r <= self.max_rate)

    def assert_valid_conductance(self, g):
        assert torch.all(0 <= g)

    def assert_valid_voltage(self, v):
        assert torch.all(self.EI <= v)
        assert torch.all(v <= self.EE)

    def weights_from_omega(self, omega):
        return torch.nn.functional.softplus(omega)

    def omega_from_weights(self, weights):
        assert torch.all(weights >= 0.0)
        return torch.log(torch.exp(weights) - 1.0)

    def weightsE(self, i):
        weightsEi = self.weights_from_omega(self._omegaE[i])
        assert torch.all(weightsEi >= 0.0)
        return weightsEi

    def weightsI(self, i):
        weightsIi = self.weights_from_omega(self._omegaI[i])
        assert torch.all(weightsIi >= 0.0)
        return weightsIi

    def set_weightsE(self, d, val):
        assert torch.all(val >= 0.0)
        assert val.shape == self._omegaE[d].shape
        self._omegaE[d].data = self.omega_from_weights(val)

    def set_weightsI(self, d, val):
        assert torch.all(val >= 0.0)
        assert val.shape == self._omegaI[d].shape
        self._omegaI[d].data = self.omega_from_weights(val)

    def scale_weightsE(self, scale, d=None):
        if d is None:
            for d in range(self.n_dendrites):
                self.set_weightsE(d, scale * self.weightsE(d))
        else:
            self.set_weightsE(d, scale * self.weightsE(d))

    def scale_weightsI(self, scale, d=None):
        if d is None:
            for d in range(self.n_dendrites):
                self.set_weightsI(d, scale * self.weightsI(d))
        else:
            self.set_weightsI(d, scale * self.weightsI(d))

    def set_input_scale(self, d, val):
        self.input_scale[self._input_slice(d)] = val

    def init_weights(self):
        scale = 0.2
        for d, in_features in enumerate(self.in_features_per_dendrite):
            if in_features > 0:
                stdv = scale * 1.0 / math.sqrt(in_features)
                initial_weightE = torch.DoubleTensor(
                    self.in_features_per_dendrite[d], self.out_features
                ).uniform_(0, stdv) * (self.EL - self.EI) / (self.EE - self.EL)
                self._omegaE[d].data = self.omega_from_weights(initial_weightE)
                initial_weightI = torch.DoubleTensor(
                    self.in_features_per_dendrite[d], self.out_features
                ).uniform_(0, stdv)
                self._omegaI[d].data = self.omega_from_weights(initial_weightI)

    def _input_slice(self, d):
        if d == 0:
            return slice(0, self.in_features_per_dendrite[0])
        else:
            return slice(
                sum(self.in_features_per_dendrite[:d]),
                sum(self.in_features_per_dendrite[: d + 1]),
            )

    def dendritic_input(self, u_in, d):
        r_in = self.input_scale * self.f(u_in)
        return r_in[:, self._input_slice(d)]

    def forward(self, u_in):

        self.assert_valid_voltage(u_in)
        assert u_in.shape[1] == self.in_features

        g0, u0 = self._forward(u_in)

        if g0 is not None:
            self.assert_valid_conductance(g0)
        self.assert_valid_voltage(u0)
        return g0, u0

    def copy_omegaE_omegaI_from(self, other):
        assert self.n_dendrites == other.n_dendrites

        for d in range(self.n_dendrites):
            self._omegaE[d].data = other._omegaE[d].data.clone()
            self._omegaI[d].data = other._omegaI[d].data.clone()

    def compute_gff0_and_uff0(self):
        gff0 = torch.ones(1, self.out_features, dtype=torch.double) * self.gL0
        uff0 = torch.ones(1, self.out_features, dtype=torch.double) * self.EL

        self.assert_valid_conductance(gff0)
        self.assert_valid_voltage(uff0)
        return gff0, uff0

    def compute_gEd_gId(self, u_in, d):
        gEd = torch.mm(self.dendritic_input(u_in, d), self.weightsE(d))
        gId = torch.mm(self.dendritic_input(u_in, d), self.weightsI(d))
        self.assert_valid_conductance(gEd)
        self.assert_valid_conductance(gId)

        return gEd, gId

    def compute_gffd_and_uffd(self, u_in):
        gffd = torch.empty(len(u_in), self.out_features, self.n_dendrites, dtype=torch.double)
        uffd = torch.empty(len(u_in), self.out_features, self.n_dendrites, dtype=torch.double)

        assert len(self.gLd) == self.n_dendrites

        for d in range(self.n_dendrites):
            gEd, gId = self.compute_gEd_gId(u_in, d)
            gffd[:, :, d] = self.gLd[d] + gEd + gId
            # need to use cloned gffd to avoid pytorch
            # inplace-modification error when calling backward() when
            # training with backprop
            uffd[:, :, d] = (
                self.gLd[d] * self.EL + gEd * self.EE + gId * self.EI
            ) / gffd[:, :, d].clone()

        self.assert_valid_conductance(gffd)
        self.assert_valid_voltage(uffd)
        return gffd, uffd

    def compute_Iffd(self, u_in):
        Iffd = torch.empty(len(u_in), self.out_features, self.n_dendrites, dtype=torch.double)

        for d in range(self.n_dendrites):
            gEd, gId = self.compute_gEd_gId(u_in, d)
            Iffd[:, :, d] = gEd - gId

        return Iffd

    def sample(self, g0, u0):
        raise NotImplementedError()

    def _forward(self, u_in):
        raise NotImplementedError()

    def energy_target(self, u0_target, g0, u0):
        raise NotImplementedError()

    def loss_target(self, u0_target, g0, u0):
        raise NotImplementedError()

    def compute_grad_manual_target(self, u0_target, g0, u0, u_in):
        raise NotImplementedError()

    def apply_grad_weights(self, lr):
        for d in range(self.n_dendrites):
            self._omegaE[d].data -= lr * self._omegaE[d]._grad
            assert torch.all(self.weightsE(d) > 0.0)
            self._omegaI[d].data -= lr * self._omegaI[d]._grad
            assert torch.all(self.weightsI(d) > 0.0)
