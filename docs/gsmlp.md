# GSMLP reference implementation

## Scope and authorship

GSMLP uses a multilayer perceptron with input-dependent Gaussian synapses. Scientific model and original algorithms: Juan Luis Crespo-Mariño, Tecnológico de Costa Rica. Original Python implementation: Aníbal Sancho-Theoduloz. Correspondence: jcrespo@itcr.ac.cr.

This distribution is a Python 3 release candidate, not a tagged public release. Training supports exactly one hidden layer. Construction and forward evaluation support multiple hidden layers; training such configurations raises `NotImplementedError`. Neurons have no additive bias in this implementation.

Foundational publication: Duro, R. J., Crespo, J. L., and Santos, J. (1999). Training Higher Order Gaussian Synapses. *Foundations and Tools for Neural Modeling*, LNCS 1606, 537–545. DOI: [10.1007/BFb0098211](https://doi.org/10.1007/BFb0098211).

## Mathematical definition

For incoming scalar activation x, a synapse contributes

```math
w(x)=A\exp(B(x-C)^2),\qquad s(x)=xw(x).
```

The distinction is essential: the network sums **x w(x)**, not just w(x). Each neuron produces

```math
a_i=\sigma\left(\sum_j s_{ij}(a_j)\right),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
```

Gaussian widths are represented by B rather than a positive standard deviation. Initial B values are nonpositive. When `correction_update_weight=True`, a positive B created by an update is reflected to -B, as in the supplied code. This is a heuristic constraint operation, not a gradient of the loss. With correction disabled, positive B is allowed but can cause exponential growth; overflow raises an exception.

For n_out outputs, the single-sample scaled-space loss is

```math
E=\frac{1}{2n_{out}}\sum_k(o_k-t_k)^2.
```

Synapse derivatives are

```math
\frac{\partial s}{\partial A}=x e^{B(x-C)^2},\quad
\frac{\partial s}{\partial B}=xA(x-C)^2e^{B(x-C)^2},\quad
\frac{\partial s}{\partial C}=-2xAB(x-C)e^{B(x-C)^2},
```

and

```math
\frac{\partial s}{\partial x}=w(x)[1+2xB(x-C)].
```

Output delta is (o_k-t_k)o_k(1-o_k)/n_out. Hidden deltas sum output deltas multiplied by the relevant derivative of s with respect to the hidden activation, then multiply by h_j(1-h_j). All parameter derivatives include the loss normalization. The implementation retains explicit scalar loops and parameter objects for continuity with the supplied Python source.

## Constructor

```python
GSMLP(name, n_in, n_out, n_hidden=None, learn_rate=None,
      momentum=0, test_size=0.2, validation_frec=10,
      porcent_error_stop=4, cross_validation=False,
      correction_update_weight=True, random_state=None)
```

| Argument | Contract |
|---|---|
| name | Converted to a string; used as a model label |
| n_in, n_out | Positive integers; booleans are rejected |
| n_hidden | Nonempty list/tuple/array of positive integers; exactly one hidden layer for training |
| learn_rate | Required: positive finite scalar or [initial, final, steps] |
| momentum | Finite nonnegative scalar; no upper bound is imposed by this implementation |
| test_size | Floating fraction strictly between 0 and 1; split must leave both sets nonempty |
| validation_frec | Positive integer; evaluate holdout every this many completed epochs |
| porcent_error_stop | Finite nonnegative percentage; 0 disables the legacy small-change condition |
| cross_validation | Must be False; legacy per-epoch resplitting is explicitly rejected |
| correction_update_weight | Boolean; reflect updated positive B values if True |
| random_state | None or integer in [0, 2**32-1] |

Learning-rate schedules use `linspace(initial, final, steps)` with positive rates, positive integer steps, and final <= initial. After the schedule ends, the last rate is reused. A one-step schedule contains the initial rate only. No default learning rate or hidden-layer width is guessed when omitted.

`random_state` seeds a local Python random generator for weight initialization and the holdout split. It does not alter global `random` state. Same-seed numerical reproducibility was tested in the recorded environment; cross-version or cross-platform bitwise identity is not guaranteed. Dates/times are not reproducible.

## Data, fitting and prediction

`train(X, Y, max_epoch)` requires finite real arrays with shapes `(n_samples, n_in)` and `(n_samples, n_out)`. A single-output target must still be two-dimensional: `Y.reshape(-1, 1)`. Inputs and targets need matching sample counts and at least two samples. The requested holdout fraction must yield nonempty partitions; scikit-learn raises for an impossible split.

The split occurs before preprocessing. Input MinMaxScaler is fitted exclusively on training inputs with range [-1,1]; target MinMaxScaler exclusively on training targets with range [0.001,0.999]. Validation data are transformed, never fitted. Out-of-training-range values are not clipped and can exceed those intervals. Constant columns follow scikit-learn's MinMaxScaler convention.

`train_indices_` and `validation_indices_` are positions in the input dataset. The added attributes store indices, not a copy of the raw dataset. The validation set is not a final independent test set if used to choose hyperparameters.

`predict(x)` accepts one finite vector of length n_in and returns a NumPy vector of length n_out in original target units. Prediction requires successful training. `_evaluate(x)` is a lower-level forward routine that expects inputs already in the intended scaled coordinate system; it does not scale or unscale. `get_error(x, target)` likewise works in scaled coordinates.

Only one successful training run is supported per model instance. Subsequent train calls raise; construct a new model for a new fit. A numerically failed run can leave partially updated weights and fitted scalers; discard that instance. No rollback is promised.

## Update, batch and stopping semantics

The legacy momentum rule is intentionally retained. For each parameter theta at update u:

```math
\theta_{u+1}=\theta_u-\eta_e g_u-\mu g_{u-1}.
```

The stored previous value is the previous raw gradient accumulator, not a recursively accumulated velocity. The previous-gradient term is not multiplied by the current learning rate. This is not the usual velocity-based momentum convention; do not interpret `momentum` as a drop-in equivalent to a framework optimizer.

`model.batch=1` (default) updates after each sample. Any positive integer batch greater than 1 is interpreted as full-batch training, preserving the supplied behavior. It does not mean that many samples per mini-batch. Full-batch accumulators are sums, not means; effective update magnitude depends on training-set size. Sample order is the order from the split and is not reshuffled each epoch.

The legacy epoch bound is inclusive. From a new model with epoch=0, `max_epoch=5` allows 6 epochs; `max_epoch=0` performs one epoch. `model.epoch` records completed epochs. `max_epoch` must be a nonnegative integer. This convention is explicitly tested and retained rather than silently changed.

Starting with the second completed epoch, the relative absolute percentage change is

```math
r_e=100\left|\frac{S_{e-1}-S_e}{S_{e-1}}\right|.
```

The counter increments when 0 < r_e < porcent_error_stop and resets otherwise. Training stops after **four consecutive qualifying changes**, not three. Both a small increase and a small decrease count because absolute change is used. Exact equality gives zero change and does not increment the counter. When the previous loss is zero, change is defined as zero if current loss is also zero and infinity otherwise, preventing division by zero. This is a training-loss-change rule, not validation-based early stopping. No best-validation checkpoint is restored.

`stop_reason_` is `epoch_limit` or `small_relative_change` after success.

History fields: `error` holds accumulated losses at each weight update; `error_epoch` holds their sums per epoch; `error_val` holds sums of scaled holdout losses every validation_frec epochs; `date_time` holds epoch timestamps. Online epoch sums are measured across changing parameters and are not the loss of a single frozen end-of-epoch model. Histories are not directly original-unit MSE metrics.

## Numerical behavior and errors

The sigmoid uses a sign-stable equivalent formula to avoid exponentiating large positive arguments. Harmless underflow in Gaussian exponentials is accepted as zero. NaN/infinite inputs and parameters are rejected. Overflow, invalid arithmetic and division-by-zero during guarded forward evaluation, training or prediction raise `FloatingPointError`; values are not silently clipped to make a run succeed. Very large finite inputs may still overflow squared distances or preprocessing and therefore raise. Saturated units may have zero gradients; stability is not a guarantee of learnability.

Public argument errors generally raise `ValueError`, unsupported architecture training `NotImplementedError`, invalid lifecycle operations `RuntimeError`, and numerical failures `FloatingPointError`. Filesystem and malformed-file exceptions propagate. Direct manual mutation of internal fields or synapse objects is not a supported fitting API.

## Persistence and CSV utilities

`model.save_model(path)` explicitly saves a successfully trained model. `load_model(path)` restores it; `load_GSMLP` is an alias. **Only load trusted pickle files**, because unpickling can execute code before type checks. Compatibility with historical pickle files or arbitrary future dependency versions is not guaranteed. Parent directories must exist. Writes can overwrite existing files and are not transactional.

`model.save_weights(path=None)` exports a long-form CSV with columns `layer,neuron,input,A,B,C` and returns the path. If no path is supplied, a portable timestamp-based filename is generated. This is a diagnostic export, not a model checkpoint or an import format.

`load_data(path)` reads a nonempty rectangular finite numeric CSV without a header; blank rows are skipped. `predict_data(model, path, output_path=None)` returns predictions for every CSV row and writes a result CSV only if output_path is explicitly provided. It no longer creates an implicit file.

## Validation scope

See `validation/VALIDATION.md` and the executed test log. The suite covers argument rejection, data dimensions, numerical extremes, lifecycle checks, reproducibility, holdout isolation, persistence, CSV, learning-rate schedule, full-batch behavior, momentum, stopping, forward consistency and gradients. Passing these tests does not establish convergence for arbitrary datasets, scientific superiority, or reproduction of historical published benchmarks.
