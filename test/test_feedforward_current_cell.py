import math
import numpy as np
import pytest
import torch

from dopp import FeedForwardCurrentCell

SEED = 1234


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

    model = FeedForwardCurrentCell(params["in_features"], params["out_features"])

    u_in = torch.Tensor([[model.EL + 10.0]])

    # hand-crafted solution
    r_in = model.f(u_in)
    Iffd_target = torch.mm(r_in, wE[0]) - torch.mm(r_in, wI[0])
    u0_target = model.EL + Iffd_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    Iffd = model.compute_Iffd(u_in)
    _, u0 = model(u_in)

    assert Iffd.shape == (1, params["out_features"], len(params["in_features"]))
    assert Iffd_target.shape == (1, params["out_features"])
    assert Iffd[0, 0, 0].tolist() == pytest.approx(Iffd_target[0, 0].tolist())

    assert u0.shape == (1, params["out_features"])
    assert u0.shape == u0_target.shape
    assert u0[0, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())

    # test energy
    u0_target = u0.clone() + 5.0
    _, u0 = model(u_in)
    g0 = torch.ones_like(u0)
    p_expected = torch.sqrt(g0 / (2 * math.pi * model.lambda_e)) * torch.exp(-g0 / (2. * model.lambda_e) * (u0_target - u0) ** 2)
    assert model.energy_target(u0_target, None, u0).item() == pytest.approx(-torch.log(p_expected).item())

    # test loss
    assert model.loss_target(u0_target, None, u0).item() == pytest.approx(0.5 * (u0_target - u0).item() ** 2)


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

    model = FeedForwardCurrentCell(params["in_features"], params["out_features"])

    u_in = torch.Tensor([[model.EL, model.EL + 5.0]])

    # hand-crafted solution
    r_in = model.f(u_in)
    Iffd_target = torch.mm(r_in, wE[0]) - torch.mm(r_in, wI[0])
    u0_target = model.EL + Iffd_target

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    Iffd = model.compute_Iffd(u_in)
    _, u0 = model(u_in)

    assert Iffd.shape == (1, 1, 1)
    assert Iffd_target.shape == (1, 1)
    assert Iffd[0, 0, 0].tolist() == pytest.approx(Iffd_target[0, 0].tolist())

    assert u0.shape == (1, 1)
    assert u0.shape == u0_target.shape
    assert u0[0, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())


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

    model = FeedForwardCurrentCell(params["in_features"], params["out_features"])

    u_in = torch.Tensor([[model.EL, model.EL + 5.0]])

    # hand-crafted solution
    r_in = model.f(u_in).reshape(1, 2, 1)
    Iffd_target = torch.Tensor(
        [
            torch.mm(r_in[:, d], wE[d]) - torch.mm(r_in[:, d], wI[d])
            for d in range(2)
        ]
    ).reshape(1, 1, 2)
    u0_target = model.EL + torch.sum(Iffd_target, dim=2)

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    Iffd = model.compute_Iffd(u_in)
    _, u0 = model(u_in)

    assert Iffd.shape == (1, 1, 2)
    assert Iffd_target.shape == (1, 1, 2)
    assert Iffd[0, 0, 0].tolist() == pytest.approx(Iffd_target[0, 0, 0].tolist())
    assert Iffd[0, 0, 1].tolist() == pytest.approx(Iffd_target[0, 0, 1].tolist())

    assert u0.shape == (1, 1)
    assert u0.shape == u0_target.shape
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

    model = FeedForwardCurrentCell(params["in_features"], params["out_features"])

    u_in = torch.Tensor([[model.EL, model.EL - 5.0, model.EL + 5.0, model.EL + 10.0]])

    # handcrafted solution
    r_in = model.f(u_in).reshape(1, 2, 2)
    Iffd_target = torch.Tensor(
        [
            torch.mm(r_in[:, d], wE[d]) - torch.mm(r_in[:, d], wI[d])
            for d in range(2)
        ]
    ).reshape(1, 1, 2)
    u0_target = model.EL + torch.sum(Iffd_target, dim=2)

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    Iffd = model.compute_Iffd(u_in)
    _, u0 = model(u_in)

    assert Iffd.shape == (1, 1, 2)
    assert Iffd_target.shape == (1, 1, 2)
    assert Iffd[0, 0, 0].tolist() == pytest.approx(Iffd_target[0, 0, 0].tolist())
    assert Iffd[0, 0, 1].tolist() == pytest.approx(Iffd_target[0, 0, 1].tolist())

    assert u0.shape == (1, 1)
    assert u0.shape == u0_target.shape
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

    model = FeedForwardCurrentCell(params["in_features"], params["out_features"])

    u_in = torch.Tensor([[model.EL, model.EL - 5.0, model.EL + 5.0, model.EL + 10.0]])

    # hand-crafted solution
    r_in = model.f(u_in).reshape(1, 2, 2)
    Iffd_target = torch.empty(1, params["out_features"], len(params["in_features"]))
    for d in range(2):
        Iffd_target[:, :, d] = torch.mm(r_in[:, d], wE[d]) - torch.mm(r_in[:, d], wI[d])
    u0_target = model.EL + torch.sum(Iffd_target, dim=2)

    # model solution
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    model.set_weightsE(1, wE[1])
    model.set_weightsI(1, wI[1])
    Iffd = model.compute_Iffd(u_in)
    _, u0 = model(u_in)

    assert Iffd.shape == (1, params["out_features"], len(params["in_features"]))
    assert Iffd_target.shape == (1, params["out_features"], len(params["in_features"]))
    assert u0.shape == (1, params["out_features"])
    assert u0.shape == u0_target.shape
    for n in range(params["out_features"]):
        assert Iffd[0, n, 0].tolist() == pytest.approx(Iffd_target[0, n, 0].tolist())
        assert Iffd[0, n, 1].tolist() == pytest.approx(Iffd_target[0, n, 1].tolist())
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

    model = FeedForwardCurrentCell(params["in_features"], params["out_features"])

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
    Iffd = model.compute_Iffd(u_in)
    _, u0 = model(u_in)

    assert Iffd.shape == (
        params["batch_size"],
        params["out_features"],
        len(params["in_features"]),
    )
    assert u0.shape == (params["batch_size"], params["out_features"])

    for trial in range(params["batch_size"]):

        # hand-crafted solution
        r_in = model.f(u_in[trial]).reshape(1, 2, 2)
        Iffd_target = torch.empty(1, params["out_features"], len(params["in_features"]))
        for d in range(2):
            Iffd_target[:, :, d] = torch.mm(r_in[:, d], wE[d]) - torch.mm(r_in[:, d], wI[d])
        u0_target = model.EL + torch.sum(Iffd_target, dim=2)

        assert Iffd_target.shape == (
            1,
            params["out_features"],
            len(params["in_features"]),
        )
        assert u0_target.shape == (1, params["out_features"])

        for n in range(params["out_features"]):
            assert Iffd[trial, n, 0].tolist() == pytest.approx(
                Iffd_target[0, n, 0].tolist()
            )
            assert Iffd[trial, n, 1].tolist() == pytest.approx(
                Iffd_target[0, n, 1].tolist()
            )

            assert u0[trial, 0].tolist() == pytest.approx(u0_target[0, 0].tolist())


def test_grad_is_identical_to_backprop_grad():
    params = {
        "seed": SEED,
        "in_features": [2, 2],
        "out_features": 3,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    model = FeedForwardCurrentCell(params["in_features"], params["out_features"])

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

    with torch.no_grad():
        _, u0_manual = model(u_in)
        model.zero_grad()
        model.compute_grad_manual_target(u0_target, None, u0_manual, u_in)
        omegaE_0_grad_manual = model._omegaE[0]._grad.clone()
        omegaI_0_grad_manual = model._omegaI[0]._grad.clone()
        omegaE_1_grad_manual = model._omegaE[1]._grad.clone()
        omegaI_1_grad_manual = model._omegaI[1]._grad.clone()

    # calculate gradient with autograd
    _, u0_bp = model(u_in)
    model.zero_grad()
    model.loss_target(u0_target, None, u0_bp).sum().backward()
    omegaE_0_grad_bp = model._omegaE[0]._grad.clone()
    omegaI_0_grad_bp = model._omegaI[0]._grad.clone()
    omegaE_1_grad_bp = model._omegaE[1]._grad.clone()
    omegaI_1_grad_bp = model._omegaI[1]._grad.clone()

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


def test_backprop_multiple_times():

    torch.manual_seed(SEED)

    model = FeedForwardCurrentCell([1], 1)
    u0_target = -60.0

    u_in = torch.DoubleTensor([[model.EL]])
    _, u0 = model(u_in)
    model.zero_grad()
    model.loss_target(u0_target, None, u0).backward()

    u_in = torch.DoubleTensor([[model.EL]])
    _, u0 = model(u_in)
    model.zero_grad()
    model.loss_target(u0_target, None, u0).backward()
