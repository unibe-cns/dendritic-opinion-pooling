import math
import torch

from .abstract_convex_cell import AbstractConvexCell


class FeedForwardCurrentCell(AbstractConvexCell):
    def __init__(self, in_features_per_dendrite, out_features):
        super().__init__(in_features_per_dendrite, out_features)

        self.gc = None

    def init_weights(self):
        scale = 2.0
        for d, in_features in enumerate(self.in_features_per_dendrite):
            if in_features > 0:
                stdv = scale * 1.0 / math.sqrt(in_features)
                initial_weightE = torch.DoubleTensor(
                    self.in_features_per_dendrite[d], self.out_features
                ).uniform_(0, stdv) * 2.5 * (self.EL - self.EI) / (self.EE - self.EL)
                self._omegaE[d].data = self.omega_from_weights(initial_weightE)
                initial_weightI = torch.DoubleTensor(
                    self.in_features_per_dendrite[d], self.out_features
                ).uniform_(0, stdv)
                self._omegaI[d].data = self.omega_from_weights(initial_weightI)

    def assert_valid_voltage(self, v):
        pass

    def _compute_u0(self, uff0, Iffd):
        return uff0 + torch.sum(Iffd, dim=2)

    def _forward(self, u_in):

        assert self.gc is None

        gff0, uff0 = self.compute_gff0_and_uff0()
        Iffd = self.compute_Iffd(u_in)
        u0 = self._compute_u0(uff0, Iffd)

        return None, u0

    def compute_grad_manual_target(self, u0_target, _, u0, u_in):

        assert self.gc is None

        self.assert_valid_voltage(u0_target)
        self.assert_valid_voltage(u0)
        self.assert_valid_voltage(u_in)

        assert u0_target.shape == u0.shape

        omegaE_grad = [
            torch.empty_like(self._omegaE[d]) for d in range(self.n_dendrites)
        ]
        omegaI_grad = [
            torch.empty_like(self._omegaI[d]) for d in range(self.n_dendrites)
        ]
        for d in range(self.n_dendrites):
            omegaE_grad_d = (u0_target - u0)
            omegaI_grad_d = -(u0_target - u0)

            omegaE_grad[d] = (
                torch.einsum(
                    "ij,ik->jk", [self.dendritic_input(u_in, d), omegaE_grad_d]
                )
                * 1.0
                / (1.0 + torch.exp(-self._omegaE[d]))
            )
            omegaI_grad[d] = (
                torch.einsum(
                    "ij,ik->jk", [self.dendritic_input(u_in, d), omegaI_grad_d]
                )
                * 1.0
                / (1.0 + torch.exp(-self._omegaI[d]))
            )

            assert not torch.all(torch.isnan(omegaE_grad[d]))
            assert not torch.all(torch.isnan(omegaI_grad[d]))

            self._omegaE[d]._grad = -omegaE_grad[d].data
            self._omegaI[d]._grad = -omegaI_grad[d].data

    def energy_target(self, u0_target, _, u0):
        return 1.0 / self.lambda_e * 1. / 2.0 * (
            u0_target - u0
        ) ** 2 + 0.5 * torch.log(2 * math.pi * self.lambda_e / torch.ones_like(u0))

    def loss_target(self, u0_target, _, u0):
        return 0.5 * (u0_target - u0) ** 2
