from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from gsmlp import GSMLP

x = np.linspace(-1, 1, 40).reshape(20, 2)
y = (0.5 + 0.2 * x[:, 0]).reshape(-1, 1)
model = GSMLP('minimal', 2, 1, [2], 1e-4, random_state=42, porcent_error_stop=0)
model.train(x, y, max_epoch=5)
print('Prediction:', model.predict([0.1, 0.2]))
print('Completed epochs:', model.epoch, 'stop reason:', model.stop_reason_)
# Optional explicit checkpoint: model.save_model('gsmlp.pkl')
