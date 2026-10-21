import numpy as np
from experiment import metrics
rng=np.random.default_rng(10)
a=rng.uniform(.1,1,(32,32)).astype(complex)
for b in [a,a*(2+3j)]:
    m=metrics(b,a)
    assert m['complex_nrmse']<1e-14
    assert abs(m['ssim']-1)<1e-12
    assert m['psnr']>190
b=a.copy();b[:16]*=.5
m=metrics(b,a)
assert 0<m['complex_nrmse']<1 and m['ssim']<.99 and np.isfinite(m['psnr'])
# Compare the uniform-window formula against an optional reference implementation.
try:
    from skimage.metrics import structural_similarity
except ImportError:print('Optional scikit-image reference unavailable; analytic tests passed')
else:
    c=np.vdot(b,a)/np.vdot(b,b)
    expected=structural_similarity(abs(a),abs(c*b),data_range=1,win_size=7,gaussian_weights=False,use_sample_covariance=True)
    assert abs(m['ssim']-expected)<1e-10,(m,expected)
    print('SSIM reference agreement passed')
print('Metric tests passed')
