"""Unit test of pool='fixed' routing (the rule-based routing ablation, organic/routing.py) on CPU with random weights
(no checkpoint, no download).

    python tests/test_fixed_route.py          (or: python -m pytest tests/)
"""
import math
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import organic.model as M  # noqa: E402
from organic.model import AnatomyQueries  # noqa: E402
from organic.routing import route_matrix  # noqa: E402


class Stub(torch.nn.Module):            # only the parts score()/attach_route() use
    def __init__(self, pool):
        super().__init__(); d = 768
        self.queries = AnatomyQueries(864, d, 16, 8, 16, layers=1, heads=12, use_pe=False)
        self.pool_mode = pool
        self.W_a, self.W_b = torch.nn.Linear(d, d, bias=False), torch.nn.Linear(d, d, bias=False)
        self.g = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, d)); self.h = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, d))
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    score = M.Organic3D.score; attach_route = M.Organic3D.attach_route


def test_fixed_route():
    torch.manual_seed(0)
    sents = ['There is hydronephrosis.', 'No pleural effusion.', 'There is a mass in the left kidney.', 'Ground glass opacity in the right upper lobe.']
    T, t = torch.randn(2, 144, 768), torch.randn(4, 768)
    fx, sn = Stub('fixed'), Stub('sentence'); sn.load_state_dict(fx.state_dict())
    tf = fx.attach_route(t, sents); assert tf.shape == (4, 768 + 144)
    assert torch.equal(sn.attach_route(t, sents), t)                       # no-op for the final model's mode
    R = route_matrix(sents, fx.queries.organ_of_query())
    # beta of the fixed model: recompute exactly as score() does
    lg = torch.einsum('vkd,sd->vsk', fx.W_b(T), fx.W_a(t)) / math.sqrt(768)
    beta = lg.masked_fill((R <= 0)[None], float('-inf')).softmax(-1)
    assert torch.all(beta[:, R <= 0] == 0) and torch.allclose(beta.sum(-1), torch.ones(2, 4))
    print('slots allowed per sentence:', R.sum(1).tolist())                # 16 kidney, 16 global, 8 L kidney, 8 R lung
    assert R.sum(1).tolist() == [16.0, 16.0, 8.0, 8.0]
    S_f, S_s = fx.score(T, tf), sn.score(T, t)
    print('scores finite:', torch.isfinite(S_f).all().item(), 'shapes', tuple(S_f.shape), tuple(S_s.shape))
    assert torch.isfinite(S_f).all() and S_f.shape == S_s.shape == (2, 4)
    print('fixed differs from learned routing:', not torch.allclose(S_f, S_s))
    assert not torch.allclose(S_f, S_s)


if __name__ == '__main__':
    test_fixed_route()
    print('PASS')
