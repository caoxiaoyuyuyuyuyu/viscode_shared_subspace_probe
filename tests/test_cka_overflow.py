import numpy as np
import sys
sys.path.insert(0, 'scripts')
from stage_b_residualized_cka import _center_gram, _cka_from_centered

def test_deep_layer_large_norm():
    # Simulate Codestral L28: n=252, d=6144, large-norm hidden states
    np.random.seed(42)
    X = np.random.randn(252, 6144).astype(np.float32) * 100
    Y = np.random.randn(252, 6144).astype(np.float32) * 100
    KX = X @ X.T
    KY = Y @ Y.T
    KX_c = _center_gram(KX)
    KY_c = _center_gram(KY)
    cka = _cka_from_centered(KX_c, KY_c)
    assert np.isfinite(cka), f"CKA not finite: {cka}"
    assert cka > 0, f"CKA=0 suggests overflow: {cka}"
    print(f"PASS: CKA={cka:.4f}")

if __name__ == '__main__':
    test_deep_layer_large_norm()
