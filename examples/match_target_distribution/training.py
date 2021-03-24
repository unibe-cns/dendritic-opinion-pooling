import numpy as np
import os
import pickle
import torch

from dopp import FeedForwardCell


def numpyfy_all_torch_tensors_in_dict(d):
    for key in d:
        try:
            d[key] = d[key].detach().numpy()
        except AttributeError:
            pass
    return d


def train(params, model, model_target, *, manual_grad=False):

    if not manual_grad:
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])

    res = {}
    res["g0"] = torch.empty(params["trials"] // params["recording_interval"])
    res["u0"] = torch.empty(params["trials"] // params["recording_interval"])
    res["u0_sample"] = torch.empty(params["trials"] // params["recording_interval"])
    res["g0_target"] = torch.empty(params["trials"] // params["recording_interval"])
    res["u0_target"] = torch.empty(params["trials"] // params["recording_interval"])
    res["u0_target_sample"] = torch.empty(
        params["trials"] // params["recording_interval"]
    )
    res["r_in"] = torch.empty(params["trials"] // params["recording_interval"], 2)
    res["r_in_noisy"] = torch.empty(params["trials"] // params["recording_interval"], 2)
    res["wEd"] = torch.empty(
        params["trials"] // params["recording_interval"], model.n_dendrites
    )
    res["wId"] = torch.empty(
        params["trials"] // params["recording_interval"], model.n_dendrites
    )

    torch.save(
        model.state_dict(), os.path.join(params["save_path"], f"checkpoint_0.pkl")
    )

    batch_size = 1
    r_in = torch.ones(batch_size, 2, dtype=torch.double)
    for i in range(params["trials"]):

        if i % 1000 == 0:
            print(f'{i + 1} / {params["trials"]}', end="\r")

        r_in.zero_()
        r_in += torch.empty(batch_size, 1, dtype=torch.double).normal_(
            params["r_mu"], params["r_sigma"]
        )
        r_in[r_in <= 0.0] = 0.001
        assert torch.all(r_in > 0.0)

        r_in_noisy = r_in.clone()
        r_in_noisy[:, 0] += torch.empty(batch_size).normal_(0.0, params["sigma_0"])
        r_in_noisy[:, 1] += torch.empty(batch_size).normal_(0.0, params["sigma_1"])
        r_in_noisy[r_in_noisy <= 0.0] = 0.001
        assert torch.all(r_in_noisy > 0.0)

        u_in = model.f_inv(r_in_noisy)
        g0, u0 = model(u_in)

        if i >= params["relative_time_silent_teacher"] * params["trials"]:
            u_in_target = model.f_inv(r_in)[:, 0].reshape(batch_size, 1)
            g0_target, u0_target = model_target(u_in_target)
            u0_target_sample = model_target.sample(g0_target, u0_target)

            model.zero_grad()
            if manual_grad:
                model.compute_grad_manual_target(u0_target_sample, g0, u0, u_in)
                model.apply_grad_weights(params["lr"])
            else:
                energy = model.energy_target(u0_target_sample, g0, u0)
                energy.sum().backward()
                optimizer.step()

        if i % params["recording_interval"] == 0:
            idx = i // params["recording_interval"]

            res["r_in"][idx] = r_in[0]
            res["r_in_noisy"][idx] = r_in_noisy[0]
            if i > params["relative_time_silent_teacher"] * params["trials"]:
                res["g0_target"][idx] = g0_target.clone()
                res["u0_target"][idx] = u0_target.clone()
                res["u0_target_sample"][idx] = u0_target_sample
            else:
                res["g0_target"][idx] = 0.0
                res["u0_target"][idx] = -70.0
                res["u0_target_sample"][idx] = -70.0

            g0, u0 = model(u_in)
            res["g0"][idx] = g0
            res["u0"][idx] = u0
            res["u0_sample"][idx] = model.sample(g0, u0)[0]

            for d in range(model.n_dendrites):
                res["wEd"][idx, d] = model.weightsE(d).clone()
                res["wId"][idx, d] = model.weightsI(d).clone()

        if (i + 1) % params["check_point_interval"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(params["save_path"], f"checkpoint_{i + 1}.pkl"),
            )

    print()

    res = numpyfy_all_torch_tensors_in_dict(res)

    return res


if __name__ == "__main__":

    params = {
        "seed": 1234,
        # "trials": 2_000_000,
        "trials": 200_000,
        "relative_time_silent_teacher": 0.05,
        "recording_interval": 100,
        "check_point_interval": 10_000,
        "lr": 0.025e-2,
        "n_dendrites": 2,
        "r_mu": 1.2,
        "r_sigma": 0.5 * 1e0,
        "sigma_0": 0.05 * 7.5e-1,
        "sigma_1": 0.25 * 7.5e-1,
        "save_path": "./",
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    scale_factor = 5.0

    gL = scale_factor * 0.05

    model_target = FeedForwardCell([1], 1)
    model_target.gc = None
    model_target.gL0 = gL
    model_target.gLd = torch.ones(1) * 0.1 * gL
    model_target.scale_weightsE(scale_factor * 2.5)
    model_target.scale_weightsI(scale_factor * 3.5)

    model = FeedForwardCell([1, 1], 1)
    model.gc = None
    model.gL0 = gL
    model.gLd = torch.ones(params["n_dendrites"]) * 0.1 * gL
    model.scale_weightsE(0.8 * scale_factor * 2.5)
    model.scale_weightsI(0.8 * scale_factor * 3.5)

    # start both modalities with comparable initial weights
    model.set_weightsE(1, 1.01 * model.weightsE(0))
    model.set_weightsI(1, 0.99 * model.weightsI(0))

    torch.manual_seed(params["seed"])
    res = train(params, model, model_target, manual_grad=True)

    with open(os.path.join(params["save_path"], f"params_{params['r_sigma']}.pkl"), "wb") as f:
        pickle.dump(params, f)

    with open(os.path.join(params["save_path"], f"res_{params['r_sigma']}.pkl"), "wb") as f:
        pickle.dump(res, f)
