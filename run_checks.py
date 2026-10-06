from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / 'src'), str(root / 'validation')]
required_suites = [root / 'tests' / 'test_gsmlp.py', root / 'tests' / 'test_vihon.py']
missing = [str(path.relative_to(root)) for path in required_suites if not path.is_file()]
if missing:
    raise SystemExit('Test suite publication is incomplete. Missing: ' + ', '.join(missing))

from gsmlp import GSMLP
from vihon import VIHON
import numpy as np

assert np.isfinite(GSMLP('import-check', 2, 1, [2], 1e-4, random_state=42)._evaluate([0.1, 0.2])).all()
assert np.isfinite(VIHON('import-check', 2, 1, [2, 2], 1e-4, random_state=42)._evaluate(np.zeros((2, 3)))).all()
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root / 'tests')))
if not result.wasSuccessful():
    raise SystemExit(1)

from audit_gradients_gsmlp import audit as audit_gsmlp
from audit_gradients_vihon import audit as audit_vihon

for name, function in [('gsmlp', audit_gsmlp), ('vihon', audit_vihon)]:
    rows = function()
    if not rows.passed.all():
        raise SystemExit(name + ': gradient audit failed')
    rows.to_csv(root / 'validation' / ('gradient_check_' + name + '.csv'), index=False)
    print(name, 'comparisons:', len(rows), 'passed:', int(rows.passed.sum()), 'max error:', rows.absolute_error.max())
print('Combined unittest methods:', result.testsRun)
