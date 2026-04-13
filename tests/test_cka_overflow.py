import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from stage_b_residualized_cka import _center_gram, _cka_from_centered

def test_deep_layer_codestral_scale():
    np.random.seed(42)
    X = (np.random.randn(252, 6144) * 10).astype(np.float32)
    Y = (np.random.randn(252, 6144) * 10).astype(np.float32)
    KX = X @ X.T
    KY = Y @ Y.T
    KX_c = _center_gram(KX)
    KY_c = _center_gram(KY)
    cka = _cka_from_centered(KX_c, KY_c)
    assert np.isfinite(cka), f'CKA not finite: {cka}'
    assert cka > 0.0, f'CKA=0 suggests overflow: {cka}'
    print(f'PASS deep_codestral: CKA={cka:.4f}')

def test_extreme_large_norm():
    np.random.seed(0)
    X = (np.random.randn(252, 6144) * 100).astype(np.float32)
    Y = (np.random.randn(252, 6144) * 100).astype(np.float32)
    KX = X @ X.T
    KY = Y @ Y.T
    KX_c = _center_gram(KX)
    KY_c = _center_gram(KY)
    cka = _cka_from_centered(KX_c, KY_c)
    assert np.isfinite(cka), f'CKA not finite: {cka}'
    assert cka > 0.0, f'CKA=0 suggests overflow: {cka}'
    print(f'PASS extreme: CKA={cka:.4f}')

if __name__ == '__main__':
    test_deep_layer_codestral_scale()
    test_extreme_large_norm()
