import numpy as np
import pytest
import torch

SEED = 1234


from dopp import DynamicFeedForwardCell, FeedForwardCell


def test_single_dendrite_single_input_single_output_single_trial():
    """
    Test that dynamic feedforward cell converges to stationary solution.
    """

    params = {
        "seed": SEED,
        "in_features": [1],
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = torch.Tensor([[[1.2]]])
    wI = torch.Tensor([[[0.7]]])

    model_expected = FeedForwardCell(params["in_features"], params["out_features"])
    u_in = torch.Tensor([[model_expected.EL + 10.0]])

    model_expected.set_weightsE(0, wE[0])
    model_expected.set_weightsI(0, wI[0])
    g0_expected, u0_expected = model_expected(u_in)

    model = DynamicFeedForwardCell(params["in_features"], params["out_features"])

    # model solution
    n_steps = 500
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    with torch.no_grad():
        for _ in range(n_steps):
            g0, u0 = model(u_in)

    assert g0.item() == pytest.approx(g0_expected.item())
    assert u0.item() == pytest.approx(u0_expected.item())


def test_initialize_somatic_potential():
    params = {
        "seed": SEED,
        "in_features": [1],
        "out_features": 1,
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    wE = torch.Tensor([[[1.2]]])
    wI = torch.Tensor([[[0.7]]])

    model = DynamicFeedForwardCell(params["in_features"], params["out_features"])
    model.gL0 *= 20.0
    u_init = torch.ones(params['out_features']) * (model.EL + 2.75)
    model.initialize_somatic_potential(u_init)

    model.set_input_scale(0, 0.0)  # no input, so cell should stay at initial state
    u_in = torch.Tensor([[model.EL + 10.0]])

    # model solution
    n_steps = 2000
    model.set_weightsE(0, wE[0])
    model.set_weightsI(0, wI[0])
    with torch.no_grad():
        for step in range(n_steps):
            g0, u0 = model(u_in)
            assert u0.item() < u_init.item()  # decay to leak potential
            if step == 0:
                assert u0.item() == pytest.approx(u_init.item(), rel=0.001)  # still close to init

    assert u0.item() == pytest.approx(model.EL, rel=0.001)  # decay to leak potential
