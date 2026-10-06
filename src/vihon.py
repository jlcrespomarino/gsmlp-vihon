"""VIHON Python 3 reference implementation — release candidate.
Scientific model: Juan Luis Crespo-Mariño, Tecnológico de Costa Rica.
Original Python implementation: Aníbal Sancho-Theoduloz.
Contact: jcrespo@itcr.ac.cr. BSD-3-Clause; see LICENSE.
Corrected analytical gradients and explicit heuristics; see docs/CHANGELOG.md.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing
import pickle
import csv
import random
pend = 1
import copy
from pathlib import Path
from functools import wraps

def _integer(value, name, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if zero else 1):
        raise ValueError(name + ' must be an integer in the permitted range.')

def _number(value, name, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or (not np.isfinite(value)) or (value <= 0 if strict else value < 0):
        raise ValueError(name + ' must be finite and positive/nonnegative as required.')

def _guard(function):

    @wraps(function)
    def wrapped(*args, **kwargs):
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            return function(*args, **kwargs)
    return wrapped

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise FloatingPointError('Sigmoid input must be finite.')
    with np.errstate(under='ignore'):
        z = np.exp(-np.abs(x))
    out = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
    return float(out) if out.ndim == 0 else out

def _parameters(model):
    for li, lay in enumerate(model.layers):
        for ni, neuron in enumerate(lay.weights):
            for ii, w in enumerate(neuron):
                names = ('vA', 'vB', 'vC') if li == 0 else ('vA', 'B', 'vC') if li == 1 else ('A', 'B', 'C')
                for name in names:
                    for component in range(3) if name.startswith('v') else [None]:
                        yield (li, ni, ii, w, name, component)

def _value(w, name, component):
    return getattr(w, name) if component is None else getattr(w, name)[component]

class weight3D:

    def __init__(self, vA, vB, vC):
        self.vA = vA
        self.vB = vB
        self.vC = vC

    def get_vw(self, vx):
        return np.array([self.vA[0] * np.exp(self.vB[0] * (vx[0] - self.vC[0]) ** 2), self.vA[1] * np.exp(self.vB[1] * (vx[1] - self.vC[1]) ** 2), self.vA[2] * np.exp(self.vB[2] * (vx[2] - self.vC[2]) ** 2)])

class weight3D_1D:

    def __init__(self, vA, B, vC):
        self.vA = vA
        self.B = B
        self.vC = vC

    def get_vw(self, vx):
        return np.array(self.vA) * np.exp(self.B * np.dot(np.array(vx) - np.array(self.vC), np.array(vx) - np.array(self.vC)))

class weight1D:

    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    def get_w(self, x):
        return self.A * np.exp(self.B * (x - self.C) ** 2)

class layer:

    def __init__(self, n_in, neurons, type_layer, act=sigmoid, rng=None):
        rng = random.Random() if rng is None else rng
        self.n_in = n_in
        self.neurons = neurons
        self.weights = []
        self.act = act
        self.type_layer = type_layer
        if type_layer == 1:
            self.out = np.zeros([neurons, 3])
            for i in range(neurons * n_in):
                vA = [2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5)]
                vB = [-2.0 * rng.random(), -2.0 * rng.random(), -2.0 * rng.random()]
                vC = [2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5)]
                self.weights.append(weight3D(vA, vB, vC))
            self.weights = np.array(self.weights)
            self.weights = self.weights.reshape(neurons, n_in)
        elif type_layer == 2:
            self.out = np.zeros([neurons])
            for i in range(neurons * n_in):
                vA = [2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5)]
                B = -2.0 * rng.random()
                vC = [2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5)]
                self.weights.append(weight3D_1D(vA, B, vC))
            self.weights = np.array(self.weights)
            self.weights = self.weights.reshape(neurons, n_in)
        elif type_layer == 3:
            self.out = np.zeros(neurons)
            for i in range(neurons * n_in):
                self.weights.append(weight1D(2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5), 2.0 * (rng.random() - 0.5)))
            self.weights = np.array(self.weights)
            self.weights = self.weights.reshape(neurons, n_in)
        else:
            print('Error, no se ha introducido correctamente el tipo de capa')

    def zero_weights(self):
        if self.type_layer == 1:
            for neuron in self.weights:
                for weig in neuron:
                    weig.vA = np.zeros(3)
                    weig.vB = np.zeros(3)
                    weig.vC = np.zeros(3)
        elif self.type_layer == 2:
            for neuron in self.weights:
                for weig in neuron:
                    weig.vA = np.zeros(3)
                    weig.B = 0
                    weig.vC = np.zeros(3)
        elif self.type_layer == 3:
            for neuron in self.weights:
                for weig in neuron:
                    weig.A = 0
                    weig.B = 0
                    weig.C = 0
        else:
            print('Error, no se ha detectado el tipo de capa')

class VIHON:

    def __init__(self, name, n_in, n_out, n_hidden=None, learn_rate=None, momentum=0, test_size=0.2, validation_frec=10, porcent_error_stop=4, cross_validation=False, correction_update_weight=True, random_state=None, legacy_heuristics=True):
        _integer(n_in, 'n_in')
        _integer(n_out, 'n_out')
        if not isinstance(n_hidden, (list, tuple, np.ndarray)) or np.ndim(n_hidden) != 1 or len(n_hidden) != 2:
            raise ValueError('n_hidden must contain exactly two positive integers.')
        for width in n_hidden:
            _integer(width, 'hidden width')
        if isinstance(learn_rate, (int, float, np.integer, np.floating)) and (not isinstance(learn_rate, (bool, np.bool_))):
            _number(learn_rate, 'learn_rate', strict=True)
        else:
            if not isinstance(learn_rate, (list, tuple, np.ndarray)) or np.ndim(learn_rate) != 1 or len(learn_rate) != 3:
                raise ValueError('learn_rate must be positive or [initial,final,steps].')
            _number(learn_rate[0], 'initial rate', strict=True)
            _number(learn_rate[1], 'final rate', strict=True)
            _integer(learn_rate[2], 'schedule steps')
            if learn_rate[1] > learn_rate[0]:
                raise ValueError('Final learning rate must not exceed initial rate.')
        _number(momentum, 'momentum')
        _number(test_size, 'test_size', strict=True)
        if test_size >= 1:
            raise ValueError('test_size must be below 1.')
        _integer(validation_frec, 'validation_frec')
        _number(porcent_error_stop, 'porcent_error_stop')
        for flag in (cross_validation, correction_update_weight, legacy_heuristics):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError('Flags must be booleans.')
        if cross_validation:
            raise ValueError('Legacy per-epoch resplitting is disabled; use a fixed holdout.')
        if random_state is not None:
            _integer(random_state, 'random_state', zero=True)
            if random_state > 2 ** 32 - 1:
                raise ValueError('random_state exceeds supported range.')
        self.random_state = None if random_state is None else int(random_state)
        self.legacy_heuristics = bool(legacy_heuristics)
        self._rng = random.Random(self.random_state)
        self.heuristic_events_ = {'center_surrogate': 0, 'width_relaxation': 0, 'positive_B_reset': 0}
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
        self.cross_validation = cross_validation
        self.correction_update_weight = correction_update_weight
        self.test_size = test_size
        self.validation_frec = validation_frec
        self.porcent_error_stop = porcent_error_stop
        self._input_scaler = preprocessing.MinMaxScaler((-1, 1))
        self._target_scaler = preprocessing.MinMaxScaler((0.001, 0.999))
        if len(n_hidden) != 2:
            print('Error, no se ha introducido la informacion de capas ocultas')
        else:
            self.layers.append(layer(n_in, n_hidden[0], 1, rng=self._rng))
            self.layers.append(layer(n_hidden[0], n_hidden[1], 2, rng=self._rng))
            self.layers.append(layer(n_hidden[1], n_out, 3, rng=self._rng))
            self._layers_momentum.append(layer(n_in, n_hidden[0], 1, rng=self._rng))
            self._layers_momentum.append(layer(n_hidden[0], n_hidden[1], 2, rng=self._rng))
            self._layers_momentum.append(layer(n_hidden[1], n_out, 3, rng=self._rng))
            self._grad_layers.append(layer(n_in, n_hidden[0], 1, rng=self._rng))
            self._grad_layers.append(layer(n_hidden[0], n_hidden[1], 2, rng=self._rng))
            self._grad_layers.append(layer(n_hidden[1], n_out, 3, rng=self._rng))
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
        self._heuristic_grad_layers = copy.deepcopy(self._grad_layers)

    @_guard
    def _evaluate(self, datum):
        datum = np.asarray(datum, dtype=float)
        if datum.shape != (self.layers[0].n_in, 3) or not np.isfinite(datum).all():
            raise ValueError('Evaluation expects a finite (n_in,3) input.')
        self._assert_finite_parameters()
        if len(datum) != self.layers[0].n_in:
            print('Error la cantidad de datos no coincide con el numero de entradas de la red')
        elif len(self.layers) != 3:
            print('Se ha detectado un error en las capas ocutas.')
        else:
            i = 0
            for neuron1 in self.layers[0].weights:
                j = 0
                net_alfa = 0
                net_beta = 0
                net_cita = 0
                for weight1 in neuron1:
                    v_weight = weight1.get_vw(datum[j])
                    net_alfa += datum[j][0] * v_weight[0]
                    net_beta += datum[j][1] * v_weight[1]
                    net_cita += datum[j][2] * v_weight[2]
                    j += 1
                self.layers[0].out[i][0] = self.layers[0].act(net_alfa)
                self.layers[0].out[i][1] = self.layers[0].act(net_beta)
                self.layers[0].out[i][2] = self.layers[0].act(net_cita)
                i += 1
            i = 0
            for neuron1 in self.layers[1].weights:
                j = 0
                net = 0
                for weight1 in neuron1:
                    net += np.dot(self.layers[0].out[j], weight1.get_vw(self.layers[0].out[j]))
                    j += 1
                self.layers[1].out[i] = self.layers[1].act(net)
                i += 1
            i = 0
            for neuron1 in self.layers[2].weights:
                j = 0
                net = 0
                for weight1 in neuron1:
                    net += self.layers[1].out[j] * weight1.get_w(self.layers[1].out[j])
                    j += 1
                self.layers[2].out[i] = self.layers[2].act(net)
                i += 1
        return self.layers[2].out.copy()

    @_guard
    def train(self, data_unscaled, targets_unscaled, max_epoch):
        _integer(max_epoch, 'max_epoch', zero=True)
        _integer(self.batch, 'batch')
        if self._train_started:
            raise RuntimeError('Create a new instance for each training run.')
        data_unscaled = np.asarray(data_unscaled, dtype=float)
        targets_unscaled = np.asarray(targets_unscaled, dtype=float)
        if data_unscaled.ndim != 3 or data_unscaled.shape[1:] != (self.layers[0].n_in, 3):
            raise ValueError('Inputs must have shape (n_samples,n_in,3).')
        if targets_unscaled.ndim != 2 or targets_unscaled.shape != (len(data_unscaled), len(self.layers[-1].out)) or len(data_unscaled) < 2:
            raise ValueError('Targets must have shape (n_samples,n_out), with at least two samples.')
        if not np.isfinite(data_unscaled).all() or not np.isfinite(targets_unscaled).all():
            raise ValueError('Inputs and targets must be finite.')
        if len(data_unscaled) != len(targets_unscaled):
            pass
        elif len(self.layers) != 3:
            pass
        else:
            train_indices, validation_indices = train_test_split(np.arange(len(data_unscaled)), test_size=self.test_size, random_state=self.random_state)
            self._train_started = True
            self.train_indices_ = train_indices.copy()
            self.validation_indices_ = validation_indices.copy()
            self._input_scaler.fit(np.array([[0.0, 0.0, 0.0], [256.0, 256.0, 256.0]]))
            data = np.asarray([self._input_scaler.transform(datum) for datum in data_unscaled])
            X_train = data[train_indices]
            X_test = data[validation_indices]
            y_train = self._target_scaler.fit_transform(targets_unscaled[train_indices])
            y_test = self._target_scaler.transform(targets_unscaled[validation_indices])
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
            while self.epoch <= max_epoch and epoch_const_error <= 3:
                for d in range(len(X_train)):
                    sum_error += self.get_error(X_train[d], y_train[d])
                    Delta, delta1, delta = self._accumulate_gradients(X_train[d], y_train[d])
                    batch_count += 1
                    if batch_count == self.batch:
                        self._prepare_effective_gradients()
                        batch_count = 0
                        self.error.append(sum_error)
                        sum_error_epoch += sum_error
                        sum_error = 0
                        for x in range(len(self.layers[0].weights)):
                            for y in range(len(self.layers[0].weights[x])):
                                grad_comun_exp_v0 = np.exp(self.layers[0].weights[x][y].vB[0] * (X_train[d][y][0] - self.layers[0].weights[x][y].vC[0]) ** 2)
                                grad_comun_exp_v1 = np.exp(self.layers[0].weights[x][y].vB[1] * (X_train[d][y][1] - self.layers[0].weights[x][y].vC[1]) ** 2)
                                grad_comun_exp_v2 = np.exp(self.layers[0].weights[x][y].vB[2] * (X_train[d][y][2] - self.layers[0].weights[x][y].vC[2]) ** 2)
                                if self.epoch < len(self.learn_rate):
                                    self.layers[0].weights[x][y].vA[0] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vA[0] + self.momentum * self._layers_momentum[0].weights[x][y].vA[0]
                                    self.layers[0].weights[x][y].vA[1] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vA[1] + self.momentum * self._layers_momentum[0].weights[x][y].vA[1]
                                    self.layers[0].weights[x][y].vA[2] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vA[2] + self.momentum * self._layers_momentum[0].weights[x][y].vA[2]
                                    if self.legacy_heuristics and (grad_comun_exp_v0 < 0.001 and abs(delta[x][0]) > 0.0001):
                                        self.layers[0].weights[x][y].vB[0] = 0.9 * self.layers[0].weights[x][y].vB[0]
                                        self._layers_momentum[0].weights[x][y].vB[0] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[0].weights[x][y].vB[0] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vB[0] + self.momentum * self._layers_momentum[0].weights[x][y].vB[0]
                                        self._layers_momentum[0].weights[x][y].vB[0] = self._effective_grad_layers[0].weights[x][y].vB[0]
                                    if self.layers[0].weights[x][y].vB[0] > 0:
                                        if self.correction_update_weight:
                                            self.layers[0].weights[x][y].vB[0] = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                    if self.legacy_heuristics and (grad_comun_exp_v1 < 0.001 and abs(delta[x][1]) > 0.0001):
                                        self.layers[0].weights[x][y].vB[1] = 0.9 * self.layers[0].weights[x][y].vB[1]
                                        self._layers_momentum[0].weights[x][y].vB[1] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[0].weights[x][y].vB[1] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vB[1] + self.momentum * self._layers_momentum[0].weights[x][y].vB[1]
                                        self._layers_momentum[0].weights[x][y].vB[1] = self._effective_grad_layers[0].weights[x][y].vB[1]
                                    if self.layers[0].weights[x][y].vB[1] > 0:
                                        if self.correction_update_weight:
                                            self.layers[0].weights[x][y].vB[1] = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                    if self.legacy_heuristics and (grad_comun_exp_v2 < 0.001 and abs(delta[x][2]) > 0.0001):
                                        self.layers[0].weights[x][y].vB[2] = 0.9 * self.layers[0].weights[x][y].vB[2]
                                        self._layers_momentum[0].weights[x][y].vB[2] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[0].weights[x][y].vB[2] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vB[2] + self.momentum * self._layers_momentum[0].weights[x][y].vB[2]
                                        self._layers_momentum[0].weights[x][y].vB[2] = self._effective_grad_layers[0].weights[x][y].vB[2]
                                    if self.layers[0].weights[x][y].vB[2] > 0:
                                        if self.correction_update_weight:
                                            self.layers[0].weights[x][y].vB[2] = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                    self.layers[0].weights[x][y].vC[0] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vC[0] + self.momentum * self._layers_momentum[0].weights[x][y].vC[0]
                                    self.layers[0].weights[x][y].vC[1] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vC[1] + self.momentum * self._layers_momentum[0].weights[x][y].vC[1]
                                    self.layers[0].weights[x][y].vC[2] -= self.learn_rate[self.epoch] * self._effective_grad_layers[0].weights[x][y].vC[2] + self.momentum * self._layers_momentum[0].weights[x][y].vC[2]
                                else:
                                    self.layers[0].weights[x][y].vA[0] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vA[0] + self.momentum * self._layers_momentum[0].weights[x][y].vA[0]
                                    self.layers[0].weights[x][y].vA[1] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vA[1] + self.momentum * self._layers_momentum[0].weights[x][y].vA[1]
                                    self.layers[0].weights[x][y].vA[2] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vA[2] + self.momentum * self._layers_momentum[0].weights[x][y].vA[2]
                                    if self.legacy_heuristics and (grad_comun_exp_v0 < 0.001 and abs(delta[x][0]) > 0.0001):
                                        self.layers[0].weights[x][y].vB[0] = 0.9 * self.layers[0].weights[x][y].vB[0]
                                        self._layers_momentum[0].weights[x][y].vB[0] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[0].weights[x][y].vB[0] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vB[0] + self.momentum * self._layers_momentum[0].weights[x][y].vB[0]
                                        self._layers_momentum[0].weights[x][y].vB[0] = self._effective_grad_layers[0].weights[x][y].vB[0]
                                    if self.layers[0].weights[x][y].vB[0] > 0:
                                        if self.correction_update_weight:
                                            self.layers[0].weights[x][y].vB[0] = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                    if self.legacy_heuristics and (grad_comun_exp_v1 < 0.001 and abs(delta[x][1]) > 0.0001):
                                        self.layers[0].weights[x][y].vB[1] = 0.9 * self.layers[0].weights[x][y].vB[1]
                                        self._layers_momentum[0].weights[x][y].vB[1] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[0].weights[x][y].vB[1] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vB[1] + self.momentum * self._layers_momentum[0].weights[x][y].vB[1]
                                        self._layers_momentum[0].weights[x][y].vB[1] = self._effective_grad_layers[0].weights[x][y].vB[1]
                                    if self.layers[0].weights[x][y].vB[1] > 0:
                                        if self.correction_update_weight:
                                            self.layers[0].weights[x][y].vB[1] = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                    if self.legacy_heuristics and (grad_comun_exp_v2 < 0.001 and abs(delta[x][2]) > 0.0001):
                                        self.layers[0].weights[x][y].vB[2] = 0.9 * self.layers[0].weights[x][y].vB[2]
                                        self._layers_momentum[0].weights[x][y].vB[2] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[0].weights[x][y].vB[2] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vB[2] + self.momentum * self._layers_momentum[0].weights[x][y].vB[2]
                                        self._layers_momentum[0].weights[x][y].vB[2] = self._effective_grad_layers[0].weights[x][y].vB[2]
                                    if self.layers[0].weights[x][y].vB[2] > 0:
                                        if self.correction_update_weight:
                                            self.layers[0].weights[x][y].vB[2] = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                    self.layers[0].weights[x][y].vC[0] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vC[0] + self.momentum * self._layers_momentum[0].weights[x][y].vC[0]
                                    self.layers[0].weights[x][y].vC[1] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vC[1] + self.momentum * self._layers_momentum[0].weights[x][y].vC[1]
                                    self.layers[0].weights[x][y].vC[2] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[0].weights[x][y].vC[2] + self.momentum * self._layers_momentum[0].weights[x][y].vC[2]
                                self._layers_momentum[0].weights[x][y].vA[0] = self._effective_grad_layers[0].weights[x][y].vA[0]
                                self._layers_momentum[0].weights[x][y].vA[1] = self._effective_grad_layers[0].weights[x][y].vA[1]
                                self._layers_momentum[0].weights[x][y].vA[2] = self._effective_grad_layers[0].weights[x][y].vA[2]
                                self._layers_momentum[0].weights[x][y].vC[0] = self._effective_grad_layers[0].weights[x][y].vC[0]
                                self._layers_momentum[0].weights[x][y].vC[1] = self._effective_grad_layers[0].weights[x][y].vC[1]
                                self._layers_momentum[0].weights[x][y].vC[2] = self._effective_grad_layers[0].weights[x][y].vC[2]
                        for x in range(len(self.layers[1].weights)):
                            for y in range(len(self.layers[1].weights[x])):
                                if self.epoch < len(self.learn_rate):
                                    self.layers[1].weights[x][y].vA[0] -= self.learn_rate[self.epoch] * self._effective_grad_layers[1].weights[x][y].vA[0] + self.momentum * self._layers_momentum[1].weights[x][y].vA[0]
                                    self.layers[1].weights[x][y].vA[1] -= self.learn_rate[self.epoch] * self._effective_grad_layers[1].weights[x][y].vA[1] + self.momentum * self._layers_momentum[1].weights[x][y].vA[1]
                                    self.layers[1].weights[x][y].vA[2] -= self.learn_rate[self.epoch] * self._effective_grad_layers[1].weights[x][y].vA[2] + self.momentum * self._layers_momentum[1].weights[x][y].vA[2]
                                    if self.legacy_heuristics and (np.exp(self.layers[1].weights[x][y].B * np.dot(self.layers[0].out[y] - np.asarray(self.layers[1].weights[x][y].vC), self.layers[0].out[y] - np.asarray(self.layers[1].weights[x][y].vC))) < 0.001 and abs(delta1[x]) > 0.0001):
                                        self.layers[1].weights[x][y].B = 0.9 * self.layers[1].weights[x][y].B
                                        self.layers[1].weights[x][y].vC[0] = self.layers[1].weights[x][y].vC[0]
                                        self.layers[1].weights[x][y].vC[1] = self.layers[1].weights[x][y].vC[1]
                                        self.layers[1].weights[x][y].vC[2] = self.layers[1].weights[x][y].vC[2]
                                        self._layers_momentum[1].weights[x][y].B = 0
                                        self._layers_momentum[1].weights[x][y].vC[0] = 0
                                        self._layers_momentum[1].weights[x][y].vC[1] = 0
                                        self._layers_momentum[1].weights[x][y].vC[2] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[1].weights[x][y].B -= self.learn_rate[self.epoch] * self._effective_grad_layers[1].weights[x][y].B + self.momentum * self._layers_momentum[1].weights[x][y].B
                                        self.layers[1].weights[x][y].vC[0] -= self.learn_rate[self.epoch] * self._effective_grad_layers[1].weights[x][y].vC[0] + self.momentum * self._layers_momentum[1].weights[x][y].vC[0]
                                        self.layers[1].weights[x][y].vC[1] -= self.learn_rate[self.epoch] * self._effective_grad_layers[1].weights[x][y].vC[1] + self.momentum * self._layers_momentum[1].weights[x][y].vC[1]
                                        self.layers[1].weights[x][y].vC[2] -= self.learn_rate[self.epoch] * self._effective_grad_layers[1].weights[x][y].vC[2] + self.momentum * self._layers_momentum[1].weights[x][y].vC[2]
                                        self._layers_momentum[1].weights[x][y].B = self._effective_grad_layers[1].weights[x][y].B
                                        self._layers_momentum[1].weights[x][y].vC[0] = self._effective_grad_layers[1].weights[x][y].vC[0]
                                        self._layers_momentum[1].weights[x][y].vC[1] = self._effective_grad_layers[1].weights[x][y].vC[1]
                                        self._layers_momentum[1].weights[x][y].vC[2] = self._effective_grad_layers[1].weights[x][y].vC[2]
                                    if self.layers[1].weights[x][y].B > 0:
                                        if self.correction_update_weight:
                                            self.layers[1].weights[x][y].B = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                else:
                                    self.layers[1].weights[x][y].vA[0] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[1].weights[x][y].vA[0] + self.momentum * self._layers_momentum[1].weights[x][y].vA[0]
                                    self.layers[1].weights[x][y].vA[1] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[1].weights[x][y].vA[1] + self.momentum * self._layers_momentum[1].weights[x][y].vA[1]
                                    self.layers[1].weights[x][y].vA[2] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[1].weights[x][y].vA[2] + self.momentum * self._layers_momentum[1].weights[x][y].vA[2]
                                    if self.legacy_heuristics and (np.exp(self.layers[1].weights[x][y].B * np.dot(self.layers[0].out[y] - np.asarray(self.layers[1].weights[x][y].vC), self.layers[0].out[y] - np.asarray(self.layers[1].weights[x][y].vC))) < 0.001 and abs(delta1[x]) > 0.0001):
                                        self.layers[1].weights[x][y].B = 0.9 * self.layers[1].weights[x][y].B
                                        self.layers[1].weights[x][y].vC[0] = self.layers[1].weights[x][y].vC[0]
                                        self.layers[1].weights[x][y].vC[1] = self.layers[1].weights[x][y].vC[1]
                                        self.layers[1].weights[x][y].vC[2] = self.layers[1].weights[x][y].vC[2]
                                        self._layers_momentum[1].weights[x][y].B = 0
                                        self._layers_momentum[1].weights[x][y].vC[0] = 0
                                        self._layers_momentum[1].weights[x][y].vC[1] = 0
                                        self._layers_momentum[1].weights[x][y].vC[2] = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[1].weights[x][y].B -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[1].weights[x][y].B + self.momentum * self._layers_momentum[1].weights[x][y].B
                                        self.layers[1].weights[x][y].vC[0] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[1].weights[x][y].vC[0] + self.momentum * self._layers_momentum[1].weights[x][y].vC[0]
                                        self.layers[1].weights[x][y].vC[1] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[1].weights[x][y].vC[1] + self.momentum * self._layers_momentum[1].weights[x][y].vC[1]
                                        self.layers[1].weights[x][y].vC[2] -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[1].weights[x][y].vC[2] + self.momentum * self._layers_momentum[1].weights[x][y].vC[2]
                                        self._layers_momentum[1].weights[x][y].B = self._effective_grad_layers[1].weights[x][y].B
                                        self._layers_momentum[1].weights[x][y].vC[0] = self._effective_grad_layers[1].weights[x][y].vC[0]
                                        self._layers_momentum[1].weights[x][y].vC[1] = self._effective_grad_layers[1].weights[x][y].vC[1]
                                        self._layers_momentum[1].weights[x][y].vC[2] = self._effective_grad_layers[1].weights[x][y].vC[2]
                                    if self.layers[1].weights[x][y].B > 0:
                                        if self.correction_update_weight:
                                            self.layers[1].weights[x][y].B = -0.005
                                            pass
                                            self.heuristic_events_['positive_B_reset'] += 1
                                        else:
                                            pass
                                self._layers_momentum[1].weights[x][y].vA[0] = self._effective_grad_layers[1].weights[x][y].vA[0]
                                self._layers_momentum[1].weights[x][y].vA[1] = self._effective_grad_layers[1].weights[x][y].vA[1]
                                self._layers_momentum[1].weights[x][y].vA[2] = self._effective_grad_layers[1].weights[x][y].vA[2]
                        for x in range(len(self.layers[2].weights)):
                            for y in range(len(self.layers[2].weights[x])):
                                if self.epoch < len(self.learn_rate):
                                    self.layers[2].weights[x][y].A -= self.learn_rate[self.epoch] * self._effective_grad_layers[2].weights[x][y].A + self.momentum * self._layers_momentum[2].weights[x][y].A
                                    if self.legacy_heuristics and (np.exp(self.layers[2].weights[x][y].B * (self.layers[1].out[y] - self.layers[2].weights[x][y].C) ** 2) < 0.001 and abs(Delta[x]) > 0.0001):
                                        self.layers[2].weights[x][y].B = 0.9 * self.layers[2].weights[x][y].B
                                        self.layers[2].weights[x][y].C = self.layers[2].weights[x][y].C
                                        self._layers_momentum[2].weights[x][y].B = 0
                                        self._layers_momentum[2].weights[x][y].C = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[2].weights[x][y].B -= self.learn_rate[self.epoch] * self._effective_grad_layers[2].weights[x][y].B + self.momentum * self._layers_momentum[2].weights[x][y].B
                                        self.layers[2].weights[x][y].C -= self.learn_rate[self.epoch] * self._effective_grad_layers[2].weights[x][y].C + self.momentum * self._layers_momentum[2].weights[x][y].C
                                        self._layers_momentum[2].weights[x][y].B = self._effective_grad_layers[2].weights[x][y].B
                                        self._layers_momentum[2].weights[x][y].C = self._effective_grad_layers[2].weights[x][y].C
                                else:
                                    self.layers[2].weights[x][y].A -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[2].weights[x][y].A + self.momentum * self._layers_momentum[2].weights[x][y].A
                                    if self.legacy_heuristics and (np.exp(self.layers[2].weights[x][y].B * (self.layers[1].out[y] - self.layers[2].weights[x][y].C) ** 2) < 0.001 and abs(Delta[x]) > 0.0001):
                                        self.layers[2].weights[x][y].B = 0.9 * self.layers[2].weights[x][y].B
                                        self.layers[2].weights[x][y].C = self.layers[2].weights[x][y].C
                                        self._layers_momentum[2].weights[x][y].B = 0
                                        self._layers_momentum[2].weights[x][y].C = 0
                                        self.heuristic_events_['width_relaxation'] += 1
                                    else:
                                        self.layers[2].weights[x][y].B -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[2].weights[x][y].B + self.momentum * self._layers_momentum[2].weights[x][y].B
                                        self.layers[2].weights[x][y].C -= self.learn_rate[len(self.learn_rate) - 1] * self._effective_grad_layers[2].weights[x][y].C + self.momentum * self._layers_momentum[2].weights[x][y].C
                                        self._layers_momentum[2].weights[x][y].B = self._effective_grad_layers[2].weights[x][y].B
                                        self._layers_momentum[2].weights[x][y].C = self._effective_grad_layers[2].weights[x][y].C
                                self._layers_momentum[2].weights[x][y].A = self._effective_grad_layers[2].weights[x][y].A
                        for lay in self._effective_grad_layers:
                            lay.zero_weights()
                        for lay in self._grad_layers + self._heuristic_grad_layers:
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
            if epoch_const_error > 3:
                pass
        self._assert_finite_parameters()
        self._trained = True
        self.stop_reason_ = 'small_relative_change' if epoch_const_error > 3 else 'epoch_limit'
        return self

    @_guard
    def _accumulate_gradients(self, datum, target):
        X_train = np.asarray([datum])
        y_train = np.asarray([target])
        d = 0
        Delta = []
        for s in range(len(self.layers[2].out)):
            Delta.append((self.layers[2].out[s] - y_train[d][s]) * self.layers[2].out[s] * (1 - self.layers[2].out[s]) / len(y_train[d]))
        for i in range(len(self.layers[2].weights)):
            for s in range(len(self.layers[2].weights[i])):
                grad_comun = Delta[i] * self.layers[1].out[s] * np.exp(self.layers[2].weights[i][s].B * (self.layers[1].out[s] - self.layers[2].weights[i][s].C) ** 2)
                self._grad_layers[2].weights[i][s].A += grad_comun
                self._grad_layers[2].weights[i][s].B += grad_comun * self.layers[2].weights[i][s].A * (self.layers[1].out[s] - self.layers[2].weights[i][s].C) ** 2
                self._grad_layers[2].weights[i][s].C += -2 * grad_comun * self.layers[2].weights[i][s].A * self.layers[2].weights[i][s].B * (self.layers[1].out[s] - self.layers[2].weights[i][s].C)
        sumatorio = 0
        delta1 = []
        for s in range(len(self.layers[1].weights)):
            for i in range(len(self.layers[2].weights)):
                sumatorio += Delta[i] * self.layers[2].weights[i][s].A * np.exp(self.layers[2].weights[i][s].B * (self.layers[1].out[s] - self.layers[2].weights[i][s].C) ** 2) * (1 + 2 * self.layers[1].out[s] * self.layers[2].weights[i][s].B * (self.layers[1].out[s] - self.layers[2].weights[i][s].C))
            delta1.append(self.layers[1].out[s] * (1 - self.layers[1].out[s]) * sumatorio)
            sumatorio = 0
        for i in range(len(self.layers[1].weights)):
            for s in range(len(self.layers[1].weights[i])):
                grad_comun = delta1[i] * np.exp(self.layers[1].weights[i][s].B * ((self.layers[0].out[s][0] - self.layers[1].weights[i][s].vC[0]) ** 2 + (self.layers[0].out[s][1] - self.layers[1].weights[i][s].vC[1]) ** 2 + (self.layers[0].out[s][2] - self.layers[1].weights[i][s].vC[2]) ** 2))
                self._grad_layers[1].weights[i][s].vA[0] += grad_comun * self.layers[0].out[s][0]
                self._grad_layers[1].weights[i][s].vA[1] += grad_comun * self.layers[0].out[s][1]
                self._grad_layers[1].weights[i][s].vA[2] += grad_comun * self.layers[0].out[s][2]
                self._grad_layers[1].weights[i][s].B += grad_comun * ((self.layers[0].out[s][0] - self.layers[1].weights[i][s].vC[0]) ** 2 + (self.layers[0].out[s][1] - self.layers[1].weights[i][s].vC[1]) ** 2 + (self.layers[0].out[s][2] - self.layers[1].weights[i][s].vC[2]) ** 2) * (self.layers[0].out[s][0] * self.layers[1].weights[i][s].vA[0] + self.layers[0].out[s][1] * self.layers[1].weights[i][s].vA[1] + self.layers[0].out[s][2] * self.layers[1].weights[i][s].vA[2])
                grad_comun_vC = -2 * grad_comun * (self.layers[0].out[s][0] * self.layers[1].weights[i][s].vA[0] + self.layers[0].out[s][1] * self.layers[1].weights[i][s].vA[1] + self.layers[0].out[s][2] * self.layers[1].weights[i][s].vA[2]) * self.layers[1].weights[i][s].B
                self._grad_layers[1].weights[i][s].vC[0] += grad_comun_vC * (self.layers[0].out[s][0] - self.layers[1].weights[i][s].vC[0])
                self._grad_layers[1].weights[i][s].vC[1] += grad_comun_vC * (self.layers[0].out[s][1] - self.layers[1].weights[i][s].vC[1])
                self._grad_layers[1].weights[i][s].vC[2] += grad_comun_vC * (self.layers[0].out[s][2] - self.layers[1].weights[i][s].vC[2])
        delta = []
        sumatorio_x = 0
        sumatorio_y = 0
        sumatorio_z = 0
        for s in range(len(self.layers[0].weights)):
            for i in range(len(self.layers[1].weights)):
                sumatorio_comun = delta1[i] * np.exp(self.layers[1].weights[i][s].B * ((self.layers[0].out[s][0] - self.layers[1].weights[i][s].vC[0]) ** 2 + (self.layers[0].out[s][1] - self.layers[1].weights[i][s].vC[1]) ** 2 + (self.layers[0].out[s][2] - self.layers[1].weights[i][s].vC[2]) ** 2))
                sumatorio_x += sumatorio_comun * (self.layers[1].weights[i][s].vA[0] + 2 * (self.layers[1].weights[i][s].vA[0] * self.layers[0].out[s][0] + self.layers[1].weights[i][s].vA[1] * self.layers[0].out[s][1] + self.layers[1].weights[i][s].vA[2] * self.layers[0].out[s][2]) * self.layers[1].weights[i][s].B * (self.layers[0].out[s][0] - self.layers[1].weights[i][s].vC[0]))
                sumatorio_y += sumatorio_comun * (self.layers[1].weights[i][s].vA[1] + 2 * (self.layers[1].weights[i][s].vA[0] * self.layers[0].out[s][0] + self.layers[1].weights[i][s].vA[1] * self.layers[0].out[s][1] + self.layers[1].weights[i][s].vA[2] * self.layers[0].out[s][2]) * self.layers[1].weights[i][s].B * (self.layers[0].out[s][1] - self.layers[1].weights[i][s].vC[1]))
                sumatorio_z += sumatorio_comun * (self.layers[1].weights[i][s].vA[2] + 2 * (self.layers[1].weights[i][s].vA[0] * self.layers[0].out[s][0] + self.layers[1].weights[i][s].vA[1] * self.layers[0].out[s][1] + self.layers[1].weights[i][s].vA[2] * self.layers[0].out[s][2]) * self.layers[1].weights[i][s].B * (self.layers[0].out[s][2] - self.layers[1].weights[i][s].vC[2]))
            delta.append([self.layers[0].out[s][0] * (1 - self.layers[0].out[s][0]) * sumatorio_x, self.layers[0].out[s][1] * (1 - self.layers[0].out[s][1]) * sumatorio_y, self.layers[0].out[s][2] * (1 - self.layers[0].out[s][2]) * sumatorio_z])
            sumatorio_x = 0
            sumatorio_y = 0
            sumatorio_z = 0
        for i in range(len(self.layers[0].weights)):
            for s in range(len(self.layers[0].weights[i])):
                grad_comun_exp_x = np.exp(self.layers[0].weights[i][s].vB[0] * (X_train[d][s][0] - self.layers[0].weights[i][s].vC[0]) ** 2)
                grad_comun_exp_y = np.exp(self.layers[0].weights[i][s].vB[1] * (X_train[d][s][1] - self.layers[0].weights[i][s].vC[1]) ** 2)
                grad_comun_exp_z = np.exp(self.layers[0].weights[i][s].vB[2] * (X_train[d][s][2] - self.layers[0].weights[i][s].vC[2]) ** 2)
                grad_comun_x = delta[i][0] * grad_comun_exp_x * X_train[d][s][0]
                grad_comun_y = delta[i][1] * grad_comun_exp_y * X_train[d][s][1]
                grad_comun_z = delta[i][2] * grad_comun_exp_z * X_train[d][s][2]
                self._grad_layers[0].weights[i][s].vA[0] += grad_comun_x
                self._grad_layers[0].weights[i][s].vA[1] += grad_comun_y
                self._grad_layers[0].weights[i][s].vA[2] += grad_comun_z
                self._grad_layers[0].weights[i][s].vB[0] += grad_comun_x * self.layers[0].weights[i][s].vA[0] * (X_train[d][s][0] - self.layers[0].weights[i][s].vC[0]) ** 2
                self._grad_layers[0].weights[i][s].vB[1] += grad_comun_y * self.layers[0].weights[i][s].vA[1] * (X_train[d][s][1] - self.layers[0].weights[i][s].vC[1]) ** 2
                self._grad_layers[0].weights[i][s].vB[2] += grad_comun_z * self.layers[0].weights[i][s].vA[2] * (X_train[d][s][2] - self.layers[0].weights[i][s].vC[2]) ** 2
                self._grad_layers[0].weights[i][s].vC[0] -= 2 * grad_comun_x * self.layers[0].weights[i][s].vA[0] * (X_train[d][s][0] - self.layers[0].weights[i][s].vC[0]) * self.layers[0].weights[i][s].vB[0]
                if self.legacy_heuristics and (grad_comun_exp_x < 0.001 and abs(delta[i][0]) > 0.0001):
                    self._heuristic_grad_layers[0].weights[i][s].vC[0] -= 2 * delta[i][0] * self.layers[0].weights[i][s].vA[0] * (X_train[d][s][0] - self.layers[0].weights[i][s].vC[0]) * self.layers[0].weights[i][s].vB[0] * np.exp(-2 * (X_train[d][s][0] - self.layers[0].weights[i][s].vC[0]) ** 2) * X_train[d][s][0]
                    self._heuristic_grad_layers[0].weights[i][s].vC[0] += 2 * grad_comun_x * self.layers[0].weights[i][s].vA[0] * (X_train[d][s][0] - self.layers[0].weights[i][s].vC[0]) * self.layers[0].weights[i][s].vB[0]
                    self.heuristic_events_['center_surrogate'] += 1
                self._grad_layers[0].weights[i][s].vC[1] -= 2 * grad_comun_y * self.layers[0].weights[i][s].vA[1] * (X_train[d][s][1] - self.layers[0].weights[i][s].vC[1]) * self.layers[0].weights[i][s].vB[1]
                if self.legacy_heuristics and (grad_comun_exp_y < 0.001 and abs(delta[i][1]) > 0.0001):
                    self._heuristic_grad_layers[0].weights[i][s].vC[1] -= 2 * delta[i][1] * self.layers[0].weights[i][s].vA[1] * (X_train[d][s][1] - self.layers[0].weights[i][s].vC[1]) * self.layers[0].weights[i][s].vB[1] * np.exp(-2 * (X_train[d][s][1] - self.layers[0].weights[i][s].vC[1]) ** 2) * X_train[d][s][1]
                    self._heuristic_grad_layers[0].weights[i][s].vC[1] += 2 * grad_comun_y * self.layers[0].weights[i][s].vA[1] * (X_train[d][s][1] - self.layers[0].weights[i][s].vC[1]) * self.layers[0].weights[i][s].vB[1]
                    self.heuristic_events_['center_surrogate'] += 1
                self._grad_layers[0].weights[i][s].vC[2] -= 2 * grad_comun_z * self.layers[0].weights[i][s].vA[2] * (X_train[d][s][2] - self.layers[0].weights[i][s].vC[2]) * self.layers[0].weights[i][s].vB[2]
                if self.legacy_heuristics and (grad_comun_exp_z < 0.001 and abs(delta[i][2]) > 0.0001):
                    self._heuristic_grad_layers[0].weights[i][s].vC[2] -= 2 * delta[i][2] * self.layers[0].weights[i][s].vA[2] * (X_train[d][s][2] - self.layers[0].weights[i][s].vC[2]) * self.layers[0].weights[i][s].vB[2] * np.exp(-2 * (X_train[d][s][2] - self.layers[0].weights[i][s].vC[2]) ** 2) * X_train[d][s][2]
                    self._heuristic_grad_layers[0].weights[i][s].vC[2] += 2 * grad_comun_z * self.layers[0].weights[i][s].vA[2] * (X_train[d][s][2] - self.layers[0].weights[i][s].vC[2]) * self.layers[0].weights[i][s].vB[2]
                    self.heuristic_events_['center_surrogate'] += 1
        return (Delta, delta1, delta)

    def _assert_finite_parameters(self):
        for _, _, _, w, name, component in _parameters(self):
            if not np.isfinite(_value(w, name, component)):
                raise FloatingPointError('Non-finite synapse parameter.')

    @_guard
    def _prepare_effective_gradients(self):
        self._effective_grad_layers = copy.deepcopy(self._grad_layers)
        if self.legacy_heuristics:
            for ne, nx in zip(self._effective_grad_layers[0].weights, self._heuristic_grad_layers[0].weights):
                for we, wx in zip(ne, nx):
                    we.vC = np.asarray(we.vC) + np.asarray(wx.vC)

    @_guard
    def get_error(self, datum, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (len(self.layers[-1].out),) or not np.isfinite(target).all():
            raise ValueError('Target must be finite with n_out components.')
        out = self._evaluate(datum)
        return float(np.sum((out - target) ** 2) / (2 * len(target)))

    @_guard
    def predict(self, datum_unscaled):
        if not self._trained:
            raise RuntimeError('Train successfully before predicting.')
        datum = np.asarray(datum_unscaled, dtype=float)
        if datum.shape != (self.layers[0].n_in, 3) or not np.isfinite(datum).all():
            raise ValueError('predict requires a finite (n_in,3) input.')
        scaled = self._input_scaler.transform(datum)
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
        self._assert_finite_parameters()
        path = Path(path) if path is not None else Path('VIHONweights_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['layer', 'neuron', 'input', 'parameter', 'component', 'value'])
            for li, ni, ii, w, name, component in _parameters(self):
                writer.writerow([li, ni, ii, name, '' if component is None else component, _value(w, name, component)])
        return path

    def show_weights(self):
        for li, ni, ii, w, name, component in _parameters(self):
            print(li, ni, ii, name, component, _value(w, name, component))

def load_model(path):
    """Load trusted pickle files only; unpickling can execute code."""
    with open(path, 'rb') as stream:
        result = pickle.load(stream)
    if not isinstance(result, VIHON) or not getattr(result, '_trained', False):
        raise TypeError('Expected a trained VIHON model from this implementation.')
    result._assert_finite_parameters()
    return result
load_VIHON = load_model

def load_data(path):
    import ast
    with open(path, 'r', newline='', encoding='utf-8') as stream:
        data = np.asarray([[ast.literal_eval(cell) for cell in row] for row in csv.reader(stream) if row], dtype=float)
    if data.ndim != 3 or data.shape[-1] != 3 or data.shape[0] == 0 or (data.shape[1] == 0) or (not np.isfinite(data).all()):
        raise ValueError('CSV must contain a nonempty rectangular table of literal 3-component numeric vectors.')
    return data

def predict_data(model, path, output_path=None):
    result = np.vstack([model.predict(row) for row in load_data(path)])
    if output_path is not None:
        with open(output_path, 'w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerows(result)
    return result
