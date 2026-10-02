import torch

from loss.data_discrepancy import MixtureLikelihood


def test_gradient_pushes_membership_towards_evidence():
    lik = MixtureLikelihood(sigma0=0.1, edge_scale=1.0)
    c = torch.tensor([0.02, 2.0])          # noise-like, edge-like
    v = torch.full((2,), 0.5, requires_grad=True)
    lik(v, c).backward()
    assert v.grad[0] > 0 and v.grad[1] < 0  # decrease v on noise, increase on edges


def test_posterior_monotone_and_mask():
    lik = MixtureLikelihood(sigma0=0.1, edge_scale=1.0)
    c = torch.linspace(0, 3, 50)
    p = lik.posterior(c)
    assert torch.all(p[1:] >= p[:-1] - 1e-6)
    v = torch.rand(50)
    mask = torch.zeros(50, dtype=torch.bool)
    mask[:10] = True
    assert torch.isclose(lik(v, c, mask), lik(v[:10], c[:10]))
