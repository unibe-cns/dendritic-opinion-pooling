from scipy.integrate import solve_ivp
import torch

from .feedforward_cell import FeedForwardCell


class DynamicFeedForwardCell(FeedForwardCell):
    def __init__(self, in_features_per_dendrite, out_features):
        super().__init__(in_features_per_dendrite, out_features)

        self.dt = 0.5  # ms
        self.cm0 = 250.0  # pF

        self.u0 = torch.ones(1, self.out_features) * self.EL

    def _forward(self, u_in):

        assert len(u_in) == 1, "batch size larger than one not supported"

        gff0, uff0 = self.compute_gff0_and_uff0()
        gffd, uffd = self.compute_gffd_and_uffd(u_in)
        g0, u0 = self._compute_g0_and_u0(gff0, uff0, gffd, uffd)

        def rhs(t, u):
            return (g0[0].numpy() * (u0[0].numpy() - u)) * 1.0 / self.cm0

        res_ivp = solve_ivp(
            rhs, (0.0, self.dt), self.u0[0].numpy(), method="RK23", max_step=self.dt
        ).y[:, -1]

        self.u0[0] = torch.Tensor(res_ivp)
        return g0, self.u0

    def initialize_somatic_potential(self, u):
        self.u0[0, :] = u
