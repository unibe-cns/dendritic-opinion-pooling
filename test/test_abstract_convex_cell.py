import numpy as np
import pytest
import torch

from dopp.abstract_convex_cell import AbstractConvexCell

SEED = np.random.randint(2 ** 32)


def test_weights_from_omega_omega_from_weights():
    torch.manual_seed(SEED)

    model = AbstractConvexCell([1], 1)

    assert model.weightsE(0).item() == pytest.approx(
        model.weights_from_omega(model.omega_from_weights(model.weightsE(0))).item()
    ), SEED


def test_set_weights():
    torch.manual_seed(SEED)

    w0 = torch.Tensor([[1.618]])
    w1 = torch.Tensor([[3.141]])

    model = AbstractConvexCell([1, 1], 1)
    model.set_weightsE(0, w0)
    model.set_weightsE(1, w1)
    model.set_weightsI(0, w0)
    model.set_weightsI(1, w1)

    assert model.weightsE(0).item() == pytest.approx(w0.item())
    assert model.weightsE(1).item() == pytest.approx(w1.item())
    assert model.weightsI(0).item() == pytest.approx(w0.item())
    assert model.weightsI(1).item() == pytest.approx(w1.item())


def test_scale_weightsE():
    torch.manual_seed(SEED)

    wE = torch.Tensor([[1.618]])
    wI = torch.Tensor([[3.141]])
    scale = 0.1

    model = AbstractConvexCell([1], 1)
    model.set_weightsE(0, wE)
    model.set_weightsI(0, wI)
    model.scale_weightsE(scale)

    assert model.weightsE(0).item() == pytest.approx(scale * wE.item())
    assert model.weightsI(0).item() == pytest.approx(wI.item())


def test_scale_weightsI():
    torch.manual_seed(SEED)

    wE = torch.Tensor([[1.618]])
    wI = torch.Tensor([[3.141]])
    scale = 0.1

    model = AbstractConvexCell([1], 1)
    model.set_weightsE(0, wE)
    model.set_weightsI(0, wI)
    model.scale_weightsI(scale)

    assert model.weightsE(0).item() == pytest.approx(wE.item())
    assert model.weightsI(0).item() == pytest.approx(scale * wI.item())


def test_dendritic_input():
    torch.manual_seed(SEED)

    r_in = torch.Tensor(1, 5).normal_(std=0.1) + 5.0

    model = AbstractConvexCell([3, 2], 1)

    u_in = model.f_inv(r_in)
    assert model.dendritic_input(u_in, 0)[0].tolist() == pytest.approx(
        r_in[0, :3].tolist()
    )
    assert model.dendritic_input(u_in, 1)[0].tolist() == pytest.approx(
        r_in[0, 3 : 3 + 2].tolist()
    )


def test_copy_omegaE_omegaI_from():
    torch.manual_seed(SEED)

    model = AbstractConvexCell([1], 1)
    model_other = AbstractConvexCell([1], 1)

    assert model.weightsE(0).item() != pytest.approx(model_other.weightsE(0).item())
    assert model.weightsI(0).item() != pytest.approx(model_other.weightsI(0).item())

    model.copy_omegaE_omegaI_from(model_other)

    assert model.weightsE(0).item() == pytest.approx(model_other.weightsE(0).item())
    assert model.weightsI(0).item() == pytest.approx(model_other.weightsI(0).item())


def test_gff0_and_uff0():
    torch.manual_seed(SEED)

    gL0 = 0.333
    EL = -72.0

    model = AbstractConvexCell([1], 1)
    model.gL0 = gL0
    model.EL = EL
    gff0, uff0 = model.compute_gff0_and_uff0()

    assert gff0 == pytest.approx(gL0)
    assert uff0 == pytest.approx(EL)


def test_gffd_and_uffd_w_input():
    torch.manual_seed(SEED)

    gLd = 0.333
    EL = -68.0
    EE = -10.0
    EI = -86.0
    wE = torch.Tensor([[[1.618]]])
    wI = torch.Tensor([[[3.141]]])
    r_in = torch.Tensor(1, 1).normal_(std=0.1) + 5.0

    gffd_expected = gLd + wE * r_in + wI * r_in
    uffd_expected = (gLd * EL + wE * r_in * EE + wI * r_in * EI) / gffd_expected

    model = AbstractConvexCell([1], 1)
    model.gLd[0] = gLd
    model.EL = EL
    model.EE = EE
    model.EI = EI
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])

    u_in = model.f_inv(r_in)
    gffd, uffd = model.compute_gffd_and_uffd(u_in)

    assert gffd[0][0].tolist() == pytest.approx(gffd_expected[0][0].tolist())
    assert uffd[0][0].tolist() == pytest.approx(uffd_expected[0][0].tolist())


def test_input_scale():
    torch.manual_seed(SEED)

    scale_0 = 0.15
    scale_1 = 1.05

    r_in = torch.Tensor(1, 5).normal_(std=0.1) + 5.0

    model = AbstractConvexCell([3, 2], 1)
    model.set_input_scale(0, scale_0)
    model.set_input_scale(1, scale_1)

    u_in = model.f_inv(r_in)
    assert model.dendritic_input(u_in, 0)[0].tolist() == pytest.approx(
        (scale_0 * r_in[0, :3]).tolist()
    )
    assert model.dendritic_input(u_in, 1)[0].tolist() == pytest.approx(
        (scale_1 * r_in[0, 3 : 3 + 2]).tolist()
    )
