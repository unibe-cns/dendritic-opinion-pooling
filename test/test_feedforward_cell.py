import math
import numpy as np
import pytest
import torch

SEED = 1234

from dopp import FeedForwardCell


def test_single_dendrite_single_input_single_output_single_trial():

    params = {
        "seed": SEED,
        "in_features": [1],
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = torch.Tensor([[[1.2]]])
    wI = torch.Tensor([[[0.7]]])

    model = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor([[model.EL + 10.0]])

    # hand-crafted solution
    r_in = model.f(u_in)
    gffd_target = model.gL0 + torch.mm(r_in, wE[0]) + torch.mm(r_in, wI[0])
    uffd_target = (
        model.gL0 * model.EL
        + torch.mm(r_in, wE[0]) * model.EE
        + torch.mm(r_in, wI[0]) * model.EI
    ) / gffd_target
    g0_target = model.gL0 + model.gc * gffd_target / (gffd_target + model.gc)
    u0_target = (
        model.gL0 * model.EL
        + model.gc * gffd_target / (gffd_target + model.gc) * uffd_target
    ) / g0_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    gffd, uffd = model.compute_gffd_and_uffd(u_in)
    g0, u0 = model(u_in)

    assert gffd.shape == (1, params["out_features"], len(params["in_features"]))
    assert uffd.shape == (1, params["out_features"], len(params["in_features"]))
    assert gffd_target.shape == (1, params["out_features"])
    assert uffd_target.shape == (1, params["out_features"])
    assert gffd[0, 0, 0].tolist() == pytest.approx(gffd_target[0, 0].tolist())
    assert uffd[0, 0, 0].tolist() == pytest.approx(uffd_target[0, 0].tolist())

    assert g0.shape == (1, params["out_features"])
    assert u0.shape == (1, params["out_features"])
    assert g0.shape == g0_target.shape
    assert u0.shape == u0_target.shape
    assert g0[0, 0].tolist() == pytest.approx(g0_target[0, 0].tolist())
    assert u0[0, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())

    # test sampling
    lambda_e = 1.67
    model.lambda_e = lambda_e
    n_samples = 5000
    u0_sample = torch.empty(n_samples, model.out_features)
    for i in range(n_samples):
        u0_sample[i] = model.sample(g0, u0)

    assert torch.mean(u0_sample).item() == pytest.approx(u0_target.item(), rel=0.0001)
    assert torch.std(u0_sample).item() == pytest.approx(torch.sqrt(lambda_e / g0_target).item(), rel=0.01)

    # test energy
    u0_target = u0.clone() + 5.0
    g0, u0 = model(u_in)
    p_expected = torch.sqrt(g0 / (2 * math.pi * model.lambda_e)) * torch.exp(-g0 / (2. * model.lambda_e) * (u0_target - u0) ** 2)
    assert model.energy_target(u0_target, g0, u0).item() == pytest.approx(-torch.log(p_expected).item())

    # test loss
    assert model.loss_target(u0_target, g0, u0).item() == pytest.approx(0.5 * (u0_target - u0).item() ** 2)


def test_single_dendrite_two_inputs_single_output_single_trial():

    params = {
        "seed": SEED,
        "in_features": [2],
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = torch.Tensor([[[1.2], [0.5]]])
    wI = torch.Tensor([[[0.7], [0.4]]])

    model = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor([[model.EL, model.EL + 5.0]])

    # hand-crafted solution
    r_in = model.f(u_in)
    gffd_target = model.gLd + torch.mm(r_in, wE[0]) + torch.mm(r_in, wI[0])
    uffd_target = (
        model.gLd * model.EL
        + torch.mm(r_in, wE[0]) * model.EE
        + torch.mm(r_in, wI[0]) * model.EI
    ) / gffd_target
    g0_target = model.gL0 + model.gc * gffd_target / (gffd_target + model.gc)
    u0_target = (
        model.gL0 * model.EL
        + model.gc * gffd_target / (gffd_target + model.gc) * uffd_target
    ) / g0_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    gffd, uffd = model.compute_gffd_and_uffd(u_in)
    g0, u0 = model(u_in)

    assert gffd.shape == (1, 1, 1)
    assert uffd.shape == (1, 1, 1)
    assert gffd_target.shape == (1, 1)
    assert uffd_target.shape == (1, 1)
    assert gffd[0, 0, 0].tolist() == pytest.approx(gffd_target[0, 0].tolist())
    assert uffd[0, 0, 0].tolist() == pytest.approx(uffd_target[0, 0].tolist())

    assert g0.shape == (1, 1)
    assert u0.shape == (1, 1)
    assert g0.shape == g0_target.shape
    assert u0.shape == u0_target.shape
    assert g0[0, 0].tolist() == pytest.approx(g0_target[0, 0].tolist())


def test_two_dendrites_two_inputs_single_output_single_trial():

    params = {
        "seed": SEED,
        "in_features": [1, 1],
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = torch.Tensor([[[1.2]], [[0.5]]])
    wI = torch.Tensor([[[0.7]], [[0.4]]])

    model = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor([[model.EL, model.EL + 5.0]])

    # hand-crafted solution
    r_in = model.f(u_in).reshape(1, 2, 1)
    gffd_target = torch.Tensor(
        [
            model.gLd[d] + torch.mm(r_in[:, d], wE[d]) + torch.mm(r_in[:, d], wI[d])
            for d in range(2)
        ]
    ).reshape(1, 1, 2)
    uffd_target = (
        torch.Tensor(
            [
                model.gLd[d] * model.EL
                + torch.mm(r_in[:, d], wE[d]) * model.EE
                + torch.mm(r_in[:, d], wI[d] * model.EI)
                for d in range(2)
            ]
        ).reshape(1, 1, 2)
        / gffd_target
    )
    g0_target = model.gL0 + torch.sum(
        model.gc * gffd_target / (gffd_target + model.gc), dim=2
    )
    u0_target = (
        model.gL0 * model.EL
        + torch.sum(
            model.gc * gffd_target / (gffd_target + model.gc) * uffd_target, dim=2
        )
    ) / g0_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    gffd, uffd = model.compute_gffd_and_uffd(u_in)
    g0, u0 = model(u_in)

    assert gffd.shape == (1, 1, 2)
    assert uffd.shape == (1, 1, 2)
    assert gffd_target.shape == (1, 1, 2)
    assert uffd_target.shape == (1, 1, 2)
    assert gffd[0, 0, 0].tolist() == pytest.approx(gffd_target[0, 0, 0].tolist())
    assert gffd[0, 0, 1].tolist() == pytest.approx(gffd_target[0, 0, 1].tolist())
    assert uffd[0, 0, 0].tolist() == pytest.approx(uffd_target[0, 0, 0].tolist())
    assert uffd[0, 0, 1].tolist() == pytest.approx(uffd_target[0, 0, 1].tolist())

    assert g0.shape == (1, 1)
    assert u0.shape == (1, 1)
    assert g0.shape == g0_target.shape
    assert u0.shape == u0_target.shape
    assert g0[0, 0].tolist() == pytest.approx(g0_target[0, 0].tolist())
    assert u0[0, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())


def test_two_dendrites_two_inputs_single_output_single_trial_heterogeneous_coupling_conductance():

    params = {
        "seed": SEED,
        "in_features": [1, 1],
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = torch.Tensor([[[1.2]], [[0.5]]])
    wI = torch.Tensor([[[0.7]], [[0.4]]])
    gc = torch.Tensor([[[10.0, 5.0]]])

    model = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor([[model.EL, model.EL + 5.0]])

    # hand-crafted solution
    r_in = model.f(u_in).reshape(1, 2, 1)
    gffd_target = torch.Tensor(
        [
            model.gLd[d] + torch.mm(r_in[:, d], wE[d]) + torch.mm(r_in[:, d], wI[d])
            for d in range(2)
        ]
    ).reshape(1, 1, 2)
    uffd_target = (
        torch.Tensor(
            [
                model.gLd[d] * model.EL
                + torch.mm(r_in[:, d], wE[d]) * model.EE
                + torch.mm(r_in[:, d], wI[d] * model.EI)
                for d in range(2)
            ]
        ).reshape(1, 1, 2)
        / gffd_target
    )
    g0_target = model.gL0 + torch.sum(gc * gffd_target / (gffd_target + gc), dim=2)
    u0_target = (
        model.gL0 * model.EL
        + torch.sum(gc * gffd_target / (gffd_target + gc) * uffd_target, dim=2)
    ) / g0_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    model.gc = gc
    gffd, uffd = model.compute_gffd_and_uffd(u_in)
    g0, u0 = model(u_in)

    assert gffd.shape == (1, 1, 2)
    assert uffd.shape == (1, 1, 2)
    assert gffd_target.shape == (1, 1, 2)
    assert uffd_target.shape == (1, 1, 2)
    assert gffd[0, 0, 0].tolist() == pytest.approx(gffd_target[0, 0, 0].tolist())
    assert gffd[0, 0, 1].tolist() == pytest.approx(gffd_target[0, 0, 1].tolist())
    assert uffd[0, 0, 0].tolist() == pytest.approx(uffd_target[0, 0, 0].tolist())
    assert uffd[0, 0, 1].tolist() == pytest.approx(uffd_target[0, 0, 1].tolist())

    assert g0.shape == (1, 1)
    assert u0.shape == (1, 1)
    assert g0.shape == g0_target.shape
    assert u0.shape == u0_target.shape
    assert g0[0, 0].tolist() == pytest.approx(g0_target[0, 0].tolist())
    assert u0[0, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())


def test_two_dendrites_four_inputs_single_output_single_trial():

    params = {
        "seed": SEED,
        "in_features": [2, 2],
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = [torch.Tensor([[1.2, 1.1]]).t(), torch.Tensor([[0.5, 0.8]]).t()]
    wI = [torch.Tensor([[0.7, 0.2]]).t(), torch.Tensor([[0.4, 0.1]]).t()]

    model = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor([[model.EL, model.EL - 5.0, model.EL + 5.0, model.EL + 10.0]])

    # handcrafted solution
    r_in = model.f(u_in).reshape(1, 2, 2)
    gffd_target = torch.Tensor(
        [
            model.gLd[d] + torch.mm(r_in[:, d], wE[d]) + torch.mm(r_in[:, d], wI[d])
            for d in range(2)
        ]
    ).reshape(1, 1, 2)
    uffd_target = (
        torch.Tensor(
            [
                model.gLd[d] * model.EL
                + torch.mm(r_in[:, d], wE[d]) * model.EE
                + torch.mm(r_in[:, d], wI[d] * model.EI)
                for d in range(2)
            ]
        ).reshape(1, 1, 2)
        / gffd_target
    )
    g0_target = model.gL0 + torch.sum(
        model.gc * gffd_target / (gffd_target + model.gc), dim=2
    )
    u0_target = (
        model.gL0 * model.EL
        + torch.sum(
            model.gc * gffd_target / (gffd_target + model.gc) * uffd_target, dim=2
        )
    ) / g0_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    gffd, uffd = model.compute_gffd_and_uffd(u_in)
    g0, u0 = model(u_in)

    assert gffd.shape == (1, 1, 2)
    assert uffd.shape == (1, 1, 2)
    assert gffd_target.shape == (1, 1, 2)
    assert uffd_target.shape == (1, 1, 2)
    assert gffd[0, 0, 0].tolist() == pytest.approx(gffd_target[0, 0, 0].tolist())
    assert gffd[0, 0, 1].tolist() == pytest.approx(gffd_target[0, 0, 1].tolist())
    assert uffd[0, 0, 0].tolist() == pytest.approx(uffd_target[0, 0, 0].tolist())
    assert uffd[0, 0, 1].tolist() == pytest.approx(uffd_target[0, 0, 1].tolist())

    assert g0.shape == (1, 1)
    assert u0.shape == (1, 1)
    assert g0.shape == g0_target.shape
    assert u0.shape == u0_target.shape
    assert g0[0, 0].tolist() == pytest.approx(g0_target[0, 0].tolist())
    assert u0[0, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())


def test_two_dendrites_four_inputs_three_outputs_single_trial():

    params = {
        "seed": SEED,
        "in_features": [2, 2],
        "out_features": 3,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = [
        torch.Tensor([[1.2, 1.1], [1.2, 1.2], [1.2, 1.3]]).t(),
        torch.Tensor([[0.5, 0.8], [0.4, 0.8], [0.3, 0.8]]).t(),
    ]
    wI = [
        torch.Tensor([[0.7, 0.2], [0.7, 0.2], [0.7, 0.2]]).t(),
        torch.Tensor([[0.4, 0.1], [0.4, 0.1], [0.4, 0.1]]).t(),
    ]

    model = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor([[model.EL, model.EL - 5.0, model.EL + 5.0, model.EL + 10.0]])

    # hand-crafted solution
    r_in = model.f(u_in).reshape(1, 2, 2)
    gffd_target = torch.empty(1, params["out_features"], len(params["in_features"]))
    uffd_target = torch.empty(1, params["out_features"], len(params["in_features"]))
    for d in range(2):
        gffd_target[:, :, d] = (
            model.gLd[d] + torch.mm(r_in[:, d], wE[d]) + torch.mm(r_in[:, d], wI[d])
        )
        uffd_target[:, :, d] = (
            model.gLd[d] * model.EL
            + torch.mm(r_in[:, d], wE[d]) * model.EE
            + torch.mm(r_in[:, d], wI[d] * model.EI)
        ) / gffd_target[:, :, d]
    g0_target = model.gL0 + torch.sum(
        model.gc * gffd_target / (gffd_target + model.gc), dim=2
    )
    u0_target = (
        model.gL0 * model.EL
        + torch.sum(
            model.gc * gffd_target / (gffd_target + model.gc) * uffd_target, dim=2
        )
    ) / g0_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    gffd, uffd = model.compute_gffd_and_uffd(u_in)
    g0, u0 = model(u_in)

    assert gffd.shape == (1, params["out_features"], len(params["in_features"]))
    assert uffd.shape == (1, params["out_features"], len(params["in_features"]))
    assert gffd_target.shape == (1, params["out_features"], len(params["in_features"]))
    assert uffd_target.shape == (1, params["out_features"], len(params["in_features"]))
    assert g0.shape == (1, params["out_features"])
    assert u0.shape == (1, params["out_features"])
    assert g0.shape == g0_target.shape
    assert u0.shape == u0_target.shape
    for n in range(params["out_features"]):
        assert gffd[0, n, 0].tolist() == pytest.approx(gffd_target[0, n, 0].tolist())
        assert gffd[0, n, 1].tolist() == pytest.approx(gffd_target[0, n, 1].tolist())
        assert uffd[0, n, 0].tolist() == pytest.approx(uffd_target[0, n, 0].tolist())
        assert uffd[0, n, 1].tolist() == pytest.approx(uffd_target[0, n, 1].tolist())

        assert g0[0, 0].tolist() == pytest.approx(g0_target[0, 0].tolist())
        assert u0[0, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())


def test_two_dendrites_four_inputs_three_outputs_multiple_trials():

    params = {
        "seed": SEED,
        "in_features": [2, 2],
        "out_features": 3,
        "batch_size": 4,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = [
        torch.Tensor([[1.2, 1.1], [1.2, 1.2], [1.2, 1.3]]).t(),
        torch.Tensor([[0.5, 0.8], [0.4, 0.8], [0.3, 0.8]]).t(),
    ]
    wI = [
        torch.Tensor([[0.7, 0.2], [0.7, 0.2], [0.7, 0.2]]).t(),
        torch.Tensor([[0.4, 0.1], [0.4, 0.1], [0.4, 0.1]]).t(),
    ]

    model = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor(
        [
            [model.EL, model.EL - 5.0, model.EL + 5.0, model.EL + 10.0],
            [model.EL, model.EL - 5.0, model.EL + 5.0, model.EL + 15.0],
            [model.EL, model.EL - 5.0, model.EL - 5.0, model.EL + 10.0],
            [model.EL, model.EL + 5.0, model.EL + 5.0, model.EL + 10.0],
        ]
    )

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    gffd, uffd = model.compute_gffd_and_uffd(u_in)
    g0, u0 = model(u_in)

    assert gffd.shape == (
        params["batch_size"],
        params["out_features"],
        len(params["in_features"]),
    )
    assert uffd.shape == (
        params["batch_size"],
        params["out_features"],
        len(params["in_features"]),
    )
    assert g0.shape == (params["batch_size"], params["out_features"])
    assert u0.shape == (params["batch_size"], params["out_features"])
    for trial in range(params["batch_size"]):

        # hand-crafted solution
        r_in = model.f(u_in[trial]).reshape(1, 2, 2)
        gffd_target = torch.empty(1, params["out_features"], len(params["in_features"]))
        uffd_target = torch.empty(1, params["out_features"], len(params["in_features"]))
        for d in range(2):
            gffd_target[:, :, d] = (
                model.gLd[d] + torch.mm(r_in[:, d], wE[d]) + torch.mm(r_in[:, d], wI[d])
            )
            uffd_target[:, :, d] = (
                model.gLd[d] * model.EL
                + torch.mm(r_in[:, d], wE[d]) * model.EE
                + torch.mm(r_in[:, d], wI[d] * model.EI)
            ) / gffd_target[:, :, d]
        g0_target = model.gL0 + torch.sum(
            model.gc * gffd_target / (gffd_target + model.gc), dim=2
        )
        u0_target = (
            model.gL0 * model.EL
            + torch.sum(
                model.gc * gffd_target / (gffd_target + model.gc) * uffd_target, dim=2
            )
        ) / g0_target

        assert gffd_target.shape == (
            1,
            params["out_features"],
            len(params["in_features"]),
        )
        assert uffd_target.shape == (
            1,
            params["out_features"],
            len(params["in_features"]),
        )
        assert g0_target.shape == (1, params["out_features"])
        assert u0_target.shape == (1, params["out_features"])

        for n in range(params["out_features"]):
            assert gffd[trial, n, 0].tolist() == pytest.approx(
                gffd_target[0, n, 0].tolist()
            )
            assert gffd[trial, n, 1].tolist() == pytest.approx(
                gffd_target[0, n, 1].tolist()
            )
            assert uffd[trial, n, 0].tolist() == pytest.approx(
                uffd_target[0, n, 0].tolist()
            )
            assert uffd[trial, n, 1].tolist() == pytest.approx(
                uffd_target[0, n, 1].tolist()
            )

            assert g0[trial, 0].tolist() == pytest.approx(g0_target[0, 0].tolist())
            assert u0[trial, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())


def test_reduction_to_point_neuron():
    """check that results are independent of dendritic layout for large transfer
    conductances and small leak conductances"""

    params = {
        "seed": SEED,
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = torch.Tensor([[1.2, 0.5]]).t()
    wI = torch.Tensor([[0.7, 0.4]]).t()

    model_single = FeedForwardCell([2], params["out_features"])
    model_single.gL0 = 0.0
    model_single.gLd = torch.Tensor([0.0])
    model_single.set_weightsE(0, wE)
    model_single.set_weightsI(0, wI)

    model_double = FeedForwardCell([1, 1], params["out_features"])
    model_double.gL0 = 0.0
    model_double.gLd = torch.Tensor([0.0, 0.0])
    model_double.set_weightsE(0, wE.reshape(2, 1, 1)[0])
    model_double.set_weightsI(0, wI.reshape(2, 1, 1)[0])
    model_double.set_weightsE(1, wE.reshape(2, 1, 1)[1])
    model_double.set_weightsI(1, wI.reshape(2, 1, 1)[1])

    u_in = torch.Tensor([[model_single.EL, model_single.EL + 10.0]])

    # potentials should be different for small coupling conductances
    model_single.gc = 1.0
    model_double.gc = 1.0
    g0_single, u0_single = model_single(u_in)
    g0_double, u0_double = model_double(u_in)

    assert u0_single[0, 0].tolist() != pytest.approx(u0_double[0, 0].tolist())

    # potentials should be identical for large coupling conductances
    model_single.gc = 1_000_000.0
    model_double.gc = 1_000_000.0
    g0_single, u0_single = model_single(u_in)
    g0_double, u0_double = model_double(u_in)

    assert u0_single[0, 0].tolist() == pytest.approx(u0_double[0, 0].tolist())


def test_grad_is_identical_to_backprop_grad_point_neuron():
    params = {
        "seed": SEED,
        "in_features": [2, 2],
        "out_features": 3,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    model = FeedForwardCell(params["in_features"], params["out_features"])
    model.lambda_e = 1.67

    model.gc = None
    u0_target = torch.DoubleTensor(
        [
            [model.EL, model.EL - 5.0, model.EL + 5.0],
            [model.EL, model.EL + 5.0, model.EL - 5.0],
        ]
    )

    u_in = torch.DoubleTensor(
        [
            [model.EL, model.EL + 5.0, model.EL - 5.0, model.EL + 10.0],
            [model.EL, model.EL - 5.0, model.EL + 5.0, model.EL - 10.0],
        ]
    )

    # calculate gradient manually
    with torch.no_grad():
        g0_manual, u0_manual = model(u_in)
        model.zero_grad()
        model.compute_grad_manual_target(u0_target, g0_manual, u0_manual, u_in)
        omegaE_0_grad_manual = model._omegaE[0]._grad.clone()
        omegaI_0_grad_manual = model._omegaI[0]._grad.clone()
        omegaE_1_grad_manual = model._omegaE[1]._grad.clone()
        omegaI_1_grad_manual = model._omegaI[1]._grad.clone()

    # calculate gradient with autograd
    g0_bp, u0_bp = model(u_in)
    model.zero_grad()
    model.energy_target(u0_target, g0_bp, u0_bp).sum().backward()
    omegaE_0_grad_bp = model._omegaE[0]._grad.clone()
    omegaI_0_grad_bp = model._omegaI[0]._grad.clone()
    omegaE_1_grad_bp = model._omegaE[1]._grad.clone()
    omegaI_1_grad_bp = model._omegaI[1]._grad.clone()

    assert g0_manual[0, 0].tolist() == pytest.approx(g0_bp[0, 0].tolist())
    assert u0_manual[0, 0].tolist() == pytest.approx(u0_bp[0, 0].tolist())

    for n in range(params["out_features"]):
        assert omegaE_0_grad_manual[:, n].tolist() == pytest.approx(
            omegaE_0_grad_bp[:, n].tolist()
        )
        assert omegaI_0_grad_manual[:, n].tolist() == pytest.approx(
            omegaI_0_grad_bp[:, n].tolist()
        )
        assert omegaE_1_grad_manual[:, n].tolist() == pytest.approx(
            omegaE_1_grad_bp[:, n].tolist()
        )
        assert omegaI_1_grad_manual[:, n].tolist() == pytest.approx(
            omegaI_1_grad_bp[:, n].tolist()
        )


# def test_grad_is_identical_to_backprop_grad():
#     params = {
#         "seed": SEED,
#         "in_features": [2, 2],
#         "out_features": 3,
#     }

#     np.random.seed(params["seed"])
#     torch.manual_seed(params["seed"])

#     model = FeedForwardCell(params["in_features"], params["out_features"])

#     model.gc = torch.Tensor([2.34, 1.87])
#     u0_target = torch.DoubleTensor(
#         [
#             [model.EL, model.EL - 5.0, model.EL + 5.0],
#             [model.EL, model.EL + 5.0, model.EL - 5.0],
#         ]
#     )

#     u_in = torch.DoubleTensor(
#         [
#             [model.EL, model.EL + 5.0, model.EL - 5.0, model.EL + 10.0],
#             [model.EL, model.EL - 5.0, model.EL + 5.0, model.EL - 10.0],
#         ]
#     )

#     with torch.no_grad():
#         g0_manual, u0_manual = model(u_in)
#         model.zero_grad()
#         model.compute_grad_manual_target(u0_target, g0_manual, u0_manual, u_in)
#         omegaE_0_grad_manual = model._omegaE[0]._grad.clone()
#         omegaI_0_grad_manual = model._omegaI[0]._grad.clone()
#         omegaE_1_grad_manual = model._omegaE[1]._grad.clone()
#         omegaI_1_grad_manual = model._omegaI[1]._grad.clone()

#     # calculate gradient with autograd
#     g0_bp, u0_bp = model(u_in)
#     model.zero_grad()
#     model.energy_target(u0_target, g0_bp, u0_bp).sum().backward()
#     omegaE_0_grad_bp = model._omegaE[0]._grad.clone()
#     omegaI_0_grad_bp = model._omegaI[0]._grad.clone()
#     omegaE_1_grad_bp = model._omegaE[1]._grad.clone()
#     omegaI_1_grad_bp = model._omegaI[1]._grad.clone()

#     assert g0_manual[0, 0].tolist() == pytest.approx(g0_bp[0, 0].tolist())
#     assert u0_manual[0, 0].tolist() == pytest.approx(u0_bp[0, 0].tolist())

#     for n in range(params["out_features"]):
#         assert omegaE_0_grad_manual[:, n].tolist() == pytest.approx(
#             omegaE_0_grad_bp[:, n].tolist()
#         )
#         assert omegaI_0_grad_manual[:, n].tolist() == pytest.approx(
#             omegaI_0_grad_bp[:, n].tolist()
#         )
#         assert omegaE_1_grad_manual[:, n].tolist() == pytest.approx(
#             omegaE_1_grad_bp[:, n].tolist()
#         )
#         assert omegaI_1_grad_manual[:, n].tolist() == pytest.approx(
#             omegaI_1_grad_bp[:, n].tolist()
#         )


def test_backprop_multiple_times():

    torch.manual_seed(SEED)

    model = FeedForwardCell([1], 1)
    u0_target = -60.0

    u_in = torch.DoubleTensor([[model.EL]])
    g0, u0 = model(u_in)
    model.zero_grad()
    model.energy_target(u0_target, g0, u0).backward()

    u_in = torch.DoubleTensor([[model.EL]])
    g0, u0 = model(u_in)
    model.zero_grad()
    model.energy_target(u0_target, g0, u0).backward()
