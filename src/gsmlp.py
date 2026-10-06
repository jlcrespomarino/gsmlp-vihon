"""GSMLP Python 3 reference implementation — release candidate.
Scientific model: Juan Luis Crespo-Mariño, Tecnológico de Costa Rica.
Original Python implementation: Aníbal Sancho-Theoduloz.
Contact: jcrespo@itcr.ac.cr
BSD-3-Clause; see LICENSE. Changes from the supplied code: docs/CHANGELOG.md.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing
import pickle
import csv
import random
pend = 1
from pathlib import Path
from functools import wraps

def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if allow_zero else 1):
        raise ValueError(name + ' must be an integer in the permitted positive/nonnegative range.')

def _finite_number(value, name, minimum=None, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or (not np.isfinite(value)):
        raise ValueError(name + ' must be a finite real number.')
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(name + ' is outside the permitted range.')

def _numerical_guard(function):

    @wraps(function)
    def guarded(*args, **kwargs):
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            return function(*args, **kwargs)
    return guarded

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise FloatingPointError('Sigmoid input must be finite.')
    z = np.exp(-np.abs(x))
    result = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
    return float(result) if result.ndim == 0 else result

class weight:

    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    def get_w(self, x):
        return self.A * np.exp(self.B * (x - self.C) ** 2)

class layer:

    def __init__(self, n_in, neurons, act=sigmoid, rng=None):
        rng = random.Random() if rng is None else rng
        self.n_in = n_in
        self.neurons = neurons
        self.weights = []
        self.out = np.zeros(neurons)
        self.act = act
        for i in range(neurons * n_in):
            self.weights.append(weight(2.0 * (rng.random() - 0.5), -2.0 * rng.random(), 2.0 * (rng.random() - 0.5)))
        self.weights = np.array(self.weights)
        self.weights = self.weights.reshape(neurons, n_in)

    def zero_weights(self):
        for neuron in self.weights:
            for weig in neuron:
                weig.A = 0
                weig.B = 0
                weig.C = 0

class GSMLP:

    def __init__(self, name, n_in, n_out, n_hidden=None, learn_rate=None, momentum=0, test_size=0.2, validation_frec=10, porcent_error_stop=4, cross_validation=False, correction_update_weight=True, random_state=None):
        _positive_integer(n_in, 'n_in')
        _positive_integer(n_out, 'n_out')
        if not isinstance(n_hidden, (list, tuple, np.ndarray)) or len(n_hidden) == 0:
            raise ValueError('n_hidden must be a nonempty sequence of positive integers.')
        for size in n_hidden:
            _positive_integer(size, 'hidden layer size')
        if isinstance(learn_rate, (int, float, np.integer, np.floating)) and (not isinstance(learn_rate, (bool, np.bool_))):
            _finite_number(learn_rate, 'learn_rate', minimum=0, strict=True)
        else:
            if not isinstance(learn_rate, (list, tuple, np.ndarray)) or len(learn_rate) != 3:
                raise ValueError('learn_rate must be positive or [initial, final, steps].')
            _finite_number(learn_rate[0], 'initial learning rate', minimum=0, strict=True)
            _finite_number(learn_rate[1], 'final learning rate', minimum=0, strict=True)
            _positive_integer(learn_rate[2], 'learning rate steps')
            if learn_rate[1] > learn_rate[0]:
                raise ValueError('Final learning rate must not exceed initial rate.')
        _finite_number(momentum, 'momentum', minimum=0)
        _finite_number(test_size, 'test_size', minimum=0, strict=True)
        if test_size >= 1:
            raise ValueError('test_size must be strictly less than 1.')
        _positive_integer(validation_frec, 'validation_frec')
        _finite_number(porcent_error_stop, 'porcent_error_stop', minimum=0)
        if not isinstance(cross_validation, (bool, np.bool_)) or not isinstance(correction_update_weight, (bool, np.bool_)):
            raise ValueError('Boolean flags must be booleans.')
        if cross_validation:
            raise ValueError('Legacy per-epoch resplitting is disabled; use a fixed holdout.')
        if random_state is not None:
            _positive_integer(random_state, 'random_state', allow_zero=True)
            if random_state > 2 ** 32 - 1:
                raise ValueError('random_state exceeds the supported range.')
        self.random_state = None if random_state is None else int(random_state)
        self._rng = random.Random(self.random_state)
        self._train_started = False
        self._trained = False
        self.stop_reason_ = None
        self.name = str(name)
        self.epoch = 0
        self.error = []
        self.error_epoch = []
        self.error_val = []
        self.date_time = []
        self.layers = []
        self._grad_layers = []
        self._layers_momentum = []
        self.batch = 1
        self.momentum = momentum
        self.correction_update_weight = correction_update_weight
        self.cross_validation = cross_validation
        self.test_size = test_size
        self.validation_frec = validation_frec
        self.porcent_error_stop = porcent_error_stop
        self._input_scaler = preprocessing.MinMaxScaler((-1, 1))
        self._target_scaler = preprocessing.MinMaxScaler((0.001, 0.999))
        if len(n_hidden) < 1:
            print('Error, no se ha introducido la informacion de capas ocultas')
        elif len(n_hidden) == 1:
            self.layers.append(layer(n_in, n_hidden[0], rng=self._rng))
            self.layers.append(layer(n_hidden[0], n_out, rng=self._rng))
            self._layers_momentum.append(layer(n_in, n_hidden[0], rng=self._rng))
            self._layers_momentum.append(layer(n_hidden[0], n_out, rng=self._rng))
            self._grad_layers.append(layer(n_in, n_hidden[0], rng=self._rng))
            self._grad_layers.append(layer(n_hidden[0], n_out, rng=self._rng))
        else:
            self.layers.append(layer(n_in, n_hidden[0], rng=self._rng))
            self._layers_momentum.append(layer(n_in, n_hidden[0], rng=self._rng))
            self._grad_layers.append(layer(n_in, n_hidden[0], rng=self._rng))
            for i in range(len(n_hidden) - 1):
                self.layers.append(layer(n_hidden[i], n_hidden[i + 1], rng=self._rng))
                self._layers_momentum.append(layer(n_hidden[i], n_hidden[i + 1], rng=self._rng))
                self._grad_layers.append(layer(n_hidden[i], n_hidden[i + 1], rng=self._rng))
            self.layers.append(layer(n_hidden[len(n_hidden) - 1], n_out, rng=self._rng))
            self._layers_momentum.append(layer(n_hidden[len(n_hidden) - 1], n_out, rng=self._rng))
            self._grad_layers.append(layer(n_hidden[len(n_hidden) - 1], n_out, rng=self._rng))
        for lay in self._layers_momentum:
            lay.zero_weights()
        for lay in self._grad_layers:
            lay.zero_weights()
        if isinstance(learn_rate, (int, float, np.integer, np.floating)):
            self.learn_rate = [learn_rate]
        elif len(learn_rate) == 3:
            if learn_rate[1] > learn_rate[0]:
                print('Error, el valor final es mayor que el inicial, la estructura de creacion del indice de aprendizaje variable (learn_rate) es [incial,final,cantidad de iteraciones], favor vuelva a intentar')
            if not isinstance(learn_rate[2], (int, np.integer)) or learn_rate[2] < 0:
                print('Error, la cantidad de iteraciones debe ser un numero entero positivo, la estructura de creacion del indice de aprendizaje variable (learn_rate) es [incial,final,cantidad de iteraciones], favor vuelva a intentar')
            self.learn_rate = np.linspace(learn_rate[0], learn_rate[1], num=learn_rate[2])
        else:
            print('Error, la estructura de creacion del indice de aprendizaje variable (learn_rate) es [incial,final,cantidad de iteraciones], favor vuelva a intentar')

    @_numerical_guard
    def train(self, data_unscaled, targets_unscaled, max_epoch):
        if len(self.layers) != 2:
            raise NotImplementedError('Training supports exactly one hidden layer.')
        _positive_integer(max_epoch, 'max_epoch', allow_zero=True)
        _positive_integer(self.batch, 'batch')
        if self._train_started:
            raise RuntimeError('Create a new model for each training run; repeated train() is not supported.')
        data_unscaled = np.asarray(data_unscaled, dtype=float)
        targets_unscaled = np.asarray(targets_unscaled, dtype=float)
        if data_unscaled.ndim != 2 or data_unscaled.shape[1] != self.layers[0].n_in:
            raise ValueError('Inputs must have shape (n_samples, n_in).')
        if targets_unscaled.ndim != 2 or targets_unscaled.shape[1] != len(self.layers[-1].out):
            raise ValueError('Targets must have shape (n_samples, n_out).')
        if len(data_unscaled) != len(targets_unscaled) or len(data_unscaled) < 2:
            raise ValueError('Need at least two samples with matching input/target counts.')
        if not np.isfinite(data_unscaled).all() or not np.isfinite(targets_unscaled).all():
            raise ValueError('Inputs and targets must be finite.')
        if len(data_unscaled) != len(targets_unscaled):
            pass
        elif len(self.layers) < 2:
            pass
        else:
            data_raw = np.asarray(data_unscaled, dtype=float)
            targets_raw = np.asarray(targets_unscaled, dtype=float)
            train_indices, validation_indices = train_test_split(np.arange(len(data_raw)), test_size=self.test_size, random_state=self.random_state)
            self._train_started = True
            self.train_indices_ = train_indices.copy()
            self.validation_indices_ = validation_indices.copy()
            X_train = self._input_scaler.fit_transform(data_raw[train_indices])
            X_test = self._input_scaler.transform(data_raw[validation_indices])
            y_train = self._target_scaler.fit_transform(targets_raw[train_indices])
            y_test = self._target_scaler.transform(targets_raw[validation_indices])
            if self.batch > 1:
                self.batch = len(X_train)
            if self.batch < 1:
                self.batch = 1
            batch_count = 0
            validation_count = 0
            sum_error = 0
            backup_count = 0
            sum_error_epoch = 0
            epoch_const_error = 0
            for lay in self._grad_layers:
                lay.zero_weights()
            if len(self.layers) == 2:
                while self.epoch <= max_epoch and epoch_const_error <= 3:
                    for d in range(len(X_train)):
                        sum_error += self.get_error(X_train[d], y_train[d])
                        for k in range(len(self.layers[1].weights)):
                            for j in range(len(self.layers[1].weights[k])):
                                df_netK = (1 - self.layers[1].out[k]) * self.layers[1].out[k]
                                grad_comun = self.layers[0].out[j] * (self.layers[1].out[k] - y_train[d][k]) * df_netK * np.exp(self.layers[1].weights[k][j].B * (self.layers[0].out[j] - self.layers[1].weights[k][j].C) ** 2) / len(y_train[d])
                                self._grad_layers[1].weights[k][j].A += grad_comun
                                self._grad_layers[1].weights[k][j].B += grad_comun * self.layers[1].weights[k][j].A * (self.layers[0].out[j] - self.layers[1].weights[k][j].C) ** 2
                                self._grad_layers[1].weights[k][j].C += -2 * grad_comun * self.layers[1].weights[k][j].A * self.layers[1].weights[k][j].B * (self.layers[0].out[j] - self.layers[1].weights[k][j].C)
                        for j in range(len(self.layers[0].weights)):
                            for i in range(len(self.layers[0].weights[j])):
                                deriv_Etot_hj = 0
                                df_netH = (1 - self.layers[0].out[j]) * self.layers[0].out[j]
                                for k in range(len(self.layers[1].weights)):
                                    delta_K = self.layers[1].out[k] * (1 - self.layers[1].out[k]) * (self.layers[1].out[k] - y_train[d][k]) / len(y_train[d])
                                    deriv_Etot_hj += delta_K * self.layers[1].weights[k][j].get_w(self.layers[0].out[j]) * (1 + 2 * self.layers[0].out[j] * self.layers[1].weights[k][j].B * (self.layers[0].out[j] - self.layers[1].weights[k][j].C))
                                deriv_Etot_hnetj = df_netH * deriv_Etot_hj
                                grad_comun = X_train[d][i] * deriv_Etot_hnetj * np.exp(self.layers[0].weights[j][i].B * (X_train[d][i] - self.layers[0].weights[j][i].C) ** 2)
                                self._grad_layers[0].weights[j][i].A += grad_comun
                                self._grad_layers[0].weights[j][i].B += grad_comun * self.layers[0].weights[j][i].A * (X_train[d][i] - self.layers[0].weights[j][i].C) ** 2
                                self._grad_layers[0].weights[j][i].C += -2 * grad_comun * self.layers[0].weights[j][i].A * self.layers[0].weights[j][i].B * (X_train[d][i] - self.layers[0].weights[j][i].C)
                        batch_count += 1
                        if batch_count == self.batch:
                            batch_count = 0
                            self.error.append(sum_error)
                            sum_error_epoch += sum_error
                            sum_error = 0
                            for lay in range(len(self.layers)):
                                for x in range(len(self.layers[lay].weights)):
                                    for y in range(len(self.layers[lay].weights[x])):
                                        if self.epoch < len(self.learn_rate):
                                            self.layers[lay].weights[x][y].A -= self.learn_rate[self.epoch] * self._grad_layers[lay].weights[x][y].A + self.momentum * self._layers_momentum[lay].weights[x][y].A
                                            self.layers[lay].weights[x][y].B -= self.learn_rate[self.epoch] * self._grad_layers[lay].weights[x][y].B + self.momentum * self._layers_momentum[lay].weights[x][y].B
                                            self.layers[lay].weights[x][y].C -= self.learn_rate[self.epoch] * self._grad_layers[lay].weights[x][y].C + self.momentum * self._layers_momentum[lay].weights[x][y].C
                                            if self.layers[lay].weights[x][y].B > 0:
                                                if self.correction_update_weight:
                                                    self.layers[lay].weights[x][y].B = -self.layers[lay].weights[x][y].B
                                                    pass
                                                else:
                                                    pass
                                            self._layers_momentum[lay].weights[x][y].A = self._grad_layers[lay].weights[x][y].A
                                            self._layers_momentum[lay].weights[x][y].B = self._grad_layers[lay].weights[x][y].B
                                            self._layers_momentum[lay].weights[x][y].C = self._grad_layers[lay].weights[x][y].C
                                        else:
                                            self.layers[lay].weights[x][y].A -= self.learn_rate[len(self.learn_rate) - 1] * self._grad_layers[lay].weights[x][y].A + self.momentum * self._layers_momentum[lay].weights[x][y].A
                                            self.layers[lay].weights[x][y].B -= self.learn_rate[len(self.learn_rate) - 1] * self._grad_layers[lay].weights[x][y].B + self.momentum * self._layers_momentum[lay].weights[x][y].B
                                            self.layers[lay].weights[x][y].C -= self.learn_rate[len(self.learn_rate) - 1] * self._grad_layers[lay].weights[x][y].C + self.momentum * self._layers_momentum[lay].weights[x][y].C
                                            if self.layers[lay].weights[x][y].B > 0:
                                                if self.correction_update_weight:
                                                    self.layers[lay].weights[x][y].B = -self.layers[lay].weights[x][y].B
                                                    pass
                                                else:
                                                    pass
                                            self._layers_momentum[lay].weights[x][y].A = self._grad_layers[lay].weights[x][y].A
                                            self._layers_momentum[lay].weights[x][y].B = self._grad_layers[lay].weights[x][y].B
                                            self._layers_momentum[lay].weights[x][y].C = self._grad_layers[lay].weights[x][y].C
                            for lay in self._grad_layers:
                                lay.zero_weights()
                    self.date_time.append(datetime.today())
                    backup_count += 1
                    validation_count += 1
                    self.epoch += 1
                    if validation_count == self.validation_frec:
                        validation_count = 0
                        error_temp = 0
                        for da in range(len(X_test)):
                            error_temp += self.get_error(X_test[da], y_test[da])
                        self.error_val.append(error_temp)
                    if self.epoch >= 2:
                        porcent_error = abs((self.error_epoch[-1] - sum_error_epoch) / self.error_epoch[-1]) * 100 if self.error_epoch[-1] != 0 else 0.0 if sum_error_epoch == 0 else float('inf')
                        if porcent_error > 0 and porcent_error < self.porcent_error_stop:
                            epoch_const_error += 1
                        else:
                            epoch_const_error = 0
                    self.error_epoch.append(sum_error_epoch)
                    sum_error_epoch = 0
            elif len(self.layers) > 2:
                pass
            if epoch_const_error > 3:
                pass
        self._assert_finite_parameters()
        self._trained = True
        self.stop_reason_ = 'small_relative_change' if epoch_const_error > 3 else 'epoch_limit'
        return self

    def _assert_finite_parameters(self):
        for lay in self.layers:
            for neuron in lay.weights:
                for w in neuron:
                    if not np.isfinite([w.A, w.B, w.C]).all():
                        raise FloatingPointError('Non-finite synapse parameter.')

    @_numerical_guard
    def _evaluate(self, datum):
        value = np.asarray(datum, dtype=float)
        if value.shape != (self.layers[0].n_in,) or not np.isfinite(value).all():
            raise ValueError('Evaluation expects a finite single input vector.')
        self._assert_finite_parameters()
        for lay in self.layers:
            for i, neuron in enumerate(lay.weights):
                net = sum((value[j] * w.get_w(value[j]) for j, w in enumerate(neuron)))
                lay.out[i] = lay.act(net)
            value = lay.out
        if not np.isfinite(value).all():
            raise FloatingPointError('Non-finite network output.')
        return value.copy()

    @_numerical_guard
    def get_error(self, datum, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (len(self.layers[-1].out),) or not np.isfinite(target).all():
            raise ValueError('Target must be a finite vector with n_out components.')
        out = self._evaluate(datum)
        return float(np.sum((out - target) ** 2) / (2 * len(target)))

    @_numerical_guard
    def predict(self, datum_unscaled):
        if not self._trained:
            raise RuntimeError('Train successfully before predicting.')
        datum = np.asarray(datum_unscaled, dtype=float)
        if datum.shape != (self.layers[0].n_in,) or not np.isfinite(datum).all():
            raise ValueError('predict expects a finite single vector with n_in components.')
        scaled = self._input_scaler.transform(datum.reshape(1, -1))[0]
        result = self._target_scaler.inverse_transform(self._evaluate(scaled).reshape(1, -1))[0]
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite prediction.')
        return result

    def save_model(self, path):
        """Save a successfully trained model. Load only trusted pickle files."""
        if not self._trained:
            raise RuntimeError('Cannot save an untrained or failed model.')
        self._assert_finite_parameters()
        with open(path, 'wb') as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def save_weights(self, path=None):
        path = Path(path) if path is not None else Path('GSMLPweights_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['layer', 'neuron', 'input', 'A', 'B', 'C'])
            for li, lay in enumerate(self.layers):
                for ni, neuron in enumerate(lay.weights):
                    for ii, w in enumerate(neuron):
                        writer.writerow([li, ni, ii, w.A, w.B, w.C])
        return path

    def show_weights(self):
        for li, lay in enumerate(self.layers):
            for ni, neuron in enumerate(lay.weights):
                for ii, w in enumerate(neuron):
                    print(li, ni, ii, w.A, w.B, w.C)

def load_data(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            rows.append([float(value) for value in row])
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or len(array) == 0 or (not np.isfinite(array).all()):
        raise ValueError('CSV must contain a nonempty finite rectangular numeric table without a header.')
    return array

def predict_data(model, path, output_path=None):
    data = load_data(path)
    result = np.vstack([model.predict(row) for row in data])
    if output_path is not None:
        with open(output_path, 'w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerows(result)
    return result

def load_model(path):
    """Load trusted pickle files only; unpickling can execute code."""
    with open(path, 'rb') as stream:
        model = pickle.load(stream)
    if not isinstance(model, GSMLP) or not getattr(model, '_trained', False):
        raise TypeError('Expected a successfully trained GSMLP model from this implementation.')
    model._assert_finite_parameters()
    return model
load_GSMLP = load_model
