from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from vihon import VIHON

x = np.random.default_rng(42).uniform(0, 256, (12, 2, 3))
y = (0.2 + x[:, 0, 0] / 512).reshape(-1, 1)
for mode in (False, True):
    model = VIHON('minimal', 2, 1, [2, 2], 1e-4, random_state=42, porcent_error_stop=0, legacy_heuristics=mode)
    model.train(x, y, max_epoch=5)
    print('legacy_heuristics:', mode, 'prediction:', model.predict(x[0]))
    print('Completed epochs:', model.epoch, 'stop reason:', model.stop_reason_)
    print('Heuristic events:', model.heuristic_events_)
# Optional explicit checkpoint: model.save_model('vihon.pkl')
