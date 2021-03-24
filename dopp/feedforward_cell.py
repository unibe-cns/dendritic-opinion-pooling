import math
import torch

from .abstract_convex_cell import AbstractConvexCell


class FeedForwardCell(AbstractConvexCell):
    def __init__(self, in_features_per_dendrite, out_features):
        super().__init__(in_features_per_dendrite, out_features)

    def sample(self, g0, u0):
        return u0 + torch.sqrt(self.lambda_e / g0) * torch.empty_like(u0, dtype=torch.double).normal_()

    def _compute_g0_and_u0(self, gff0, uff0, gffd, uffd):
        g0 = torch.empty_like(gff0, dtype=torch.double)
        u0 = torch.empty_like(uff0, dtype=torch.double)
        if self.gc is None:
            g0 = gff0 + torch.sum(gffd, dim=2)
            u0 = (gff0 * uff0 + torch.sum(gffd * uffd, dim=2)) / g0
        else:
            g0 = gff0 + torch.sum(self.gc * gffd / (gffd + self.gc), dim=2)
            u0 = (
                gff0 * uff0 + torch.sum(self.gc * gffd / (gffd + self.gc) * uffd, dim=2)
            ) / g0

        return g0, u0

    def _forward(self, u_in):

        gff0, uff0 = self.compute_gff0_and_uff0()
        gffd, uffd = self.compute_gffd_and_uffd(u_in)

        g0, u0 = self._compute_g0_and_u0(gff0, uff0, gffd, uffd)

        return g0, u0

    def compute_grad_manual_target(self, u0_target, g0, u0, u_in):

        self.assert_valid_voltage(u0_target)
        self.assert_valid_conductance(g0)
        self.assert_valid_voltage(u0)
        assert u0_target.shape == u0.shape

        omegaE_grad = [
            torch.empty_like(self._omegaE[d], dtype=torch.double) for d in range(self.n_dendrites)
        ]
        omegaI_grad = [
            torch.empty_like(self._omegaI[d], dtype=torch.double) for d in range(self.n_dendrites)
        ]
        for d in range(self.n_dendrites):
            if self.gc is None:
                omegaE_grad_d = (
                    1.0 / self.lambda_e * (u0_target - u0) * (self.EE - u0)
                )
                omegaI_grad_d = (
                    1.0 / self.lambda_e * (u0_target - u0) * (self.EI - u0)
                )
            else:
                raise NotImplementedError()

            if self.gc is None:
                quad_error = -0.5 * (
                    1.0 / self.lambda_e * (u0_target - u0) ** 2
                    - 1.0 / g0
                )
            else:
                raise NotImplementedError()

            omegaE_grad_d += quad_error
            omegaI_grad_d += quad_error

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

    def energy_target(self, u0_target, g0, u0):
        return 1.0 / self.lambda_e * g0 / 2.0 * (
            u0_target - u0
        ) ** 2 + 0.5 * torch.log(2 * math.pi * self.lambda_e / g0)

    def loss_target(self, u0_target, g0, u0):
        return 0.5 * (u0_target - u0) ** 2
