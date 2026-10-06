# VIHON reference implementation

## Scope and attribution

Scientific model and original algorithms: Juan Luis Crespo-Mariño, Tecnológico de Costa Rica. Original Python implementation: Aníbal Sancho-Theoduloz. Correspondence: jcrespo@itcr.ac.cr. BSD 3-Clause.

This package is a Python 3 release candidate, not a tagged public release. It retains the supplied vector architecture while correcting analytical gradients and documenting optimizer heuristics. It is not a reproduction claim for historical published experiments.

Foundational publication: Crespo, J. L., and Duro, R. J. (2005). Considering Multidimensional Information Through Vector Neural Networks. *Computational Intelligence and Bioinspired Systems*, LNCS 3512, 17–24. DOI: [10.1007/11494669_3](https://doi.org/10.1007/11494669_3).

## Architecture and equations

A sample contains n_in vectors with three coordinates each. The first hidden layer outputs vectors, the second hidden layer outputs scalars, and the final layer outputs scalars. Exactly two hidden-layer sizes are required. There are no additive neuron biases in this implementation.

For incoming vector x, a first-layer synapse has three-component parameters A, B and C and componentwise contributions

```math
s_q(x)=x_q A_q \exp(B_q(x_q-C_q)^2),\quad q=0,1,2.
```

Each vector neuron sums incoming contributions per coordinate and applies sigmoid independently per coordinate.

A second-layer vector-to-scalar synapse uses vector A and C, scalar B and contribution

```math
s(h)=(h\cdot A)\exp(B\|h-C\|^2).
```

Each second-layer neuron sums such scalar contributions and applies sigmoid. The output synapses use the scalar contribution x A exp(B(x-C)^2), followed by sigmoid after summation.

For n_out outputs the scaled-space single-sample loss is

```math
E=\frac{1}{2n_{out}}\sum_k(o_k-t_k)^2.
```

Output delta is (o-t)o(1-o)/n_out. Backpropagation uses exact derivatives, including

```math
\nabla_h s=e^{B\|h-C\|^2}[A+2B(h-C)(h\cdot A)],
```

```math
\nabla_A s=h e^{B\|h-C\|^2},\quad
\partial_B s=(h\cdot A)\|h-C\|^2 e^{B\|h-C\|^2},\quad
\nabla_C s=-2B(h-C)(h\cdot A)e^{B\|h-C\|^2}.
```

First-layer component derivatives are the scalar Gaussian-synapse derivatives for each coordinate; the corresponding downstream derivative is multiplied by the hidden sigmoid derivative. The vB accumulators address coordinates 0, 1 and 2 separately. These analytical gradients are kept separate from heuristic modifications.

## Constructor and data contract

```python
VIHON(name, n_in, n_out, n_hidden=None, learn_rate=None,
      momentum=0, test_size=0.2, validation_frec=10,
      porcent_error_stop=4, cross_validation=False,
      correction_update_weight=True, random_state=None,
      legacy_heuristics=True)
```

| Argument | Contract |
|---|---|
| name | Converted to a string; label only |
| n_in, n_out | Positive integers, not booleans |
| n_hidden | Exactly two positive integers in a one-dimensional list/tuple/array |
| learn_rate | Required: finite positive scalar or [initial, final, steps] |
| momentum | Finite nonnegative scalar; no upper bound imposed |
| test_size | Fraction strictly between 0 and 1, leaving nonempty partitions |
| validation_frec | Positive integer |
| porcent_error_stop | Finite nonnegative percentage; zero disables the qualifying-change rule |
| cross_validation | Must be False; historical per-epoch resplitting is rejected |
| correction_update_weight | Boolean controlling historical positive-B resets |
| random_state | None or integer in [0, 2**32-1] |
| legacy_heuristics | Boolean; True retains explicitly separated heuristic policies |

Schedules use linspace(initial,final,steps), require positive rates, final<=initial and positive integer steps. A one-step schedule contains only the initial rate. The last rate is reused after schedule exhaustion. Hidden widths and rates are not guessed when omitted.

`train(X,Y,max_epoch)` expects finite real data with X shape `(n_samples,n_in,3)` and Y shape `(n_samples,n_out)`; one-output Y must remain two-dimensional. At least two samples are required. `predict(x)` accepts a single `(n_in,3)` sample and returns an `(n_out,)` NumPy vector in original target units. It does not accept flattened 3*n_in inputs or predict a whole batch implicitly.

## Preprocessing and reproducibility

Raw sample indices are split once. Coordinate input scaling is a fixed domain map x/128-1, calibrated from synthetic endpoints [0,0,0] and [256,256,256]. It is not learned from observed inputs. It handles n_in=1 without division by zero. Values outside [0,256] are not clipped.

The target MinMaxScaler is fitted only on training targets, with output range [0.001,0.999]. Holdout targets are transformed using those training-fitted parameters. Outside-range target values can exceed this interval. Constant targets follow scikit-learn's MinMaxScaler convention. The holdout is validation, not an unbiased final test set if used to select model settings.

`train_indices_` and `validation_indices_` store original sample positions, not raw data copies. `random_state` seeds a local Python random generator for initial parameters and the fixed split; it leaves global random state untouched. Numerical reproducibility was verified in the recorded environment; timestamps differ, and identity across platforms/dependency versions is not promised.

## Training lifecycle and numerical failures

A model supports one training run. `train` returns self on success. Repeated train calls raise RuntimeError to avoid refitting scalers while retaining old weights. Argument errors before fitting do not mark a run as started; data can be corrected and retried. Once the split succeeds, the instance is marked started. Failed numerical training may leave partial weights/scalers; discard that instance. No rollback is promised. Prediction and checkpoint saving require successful training.

Stable sigmoid avoids exp(large positive number). Gaussian underflow is accepted as zero. Overflow, invalid numerical arithmetic and division by zero in guarded forward evaluation, gradient accumulation, training and prediction raise FloatingPointError. NaN/infinite data and synapse parameters are rejected. Values are not silently clipped or replaced. Extremely large but finite data may still overflow squared distances or scaler arithmetic and raise. Saturation can produce zero gradients; numerical stability does not guarantee learnability. Direct mutation of internal synapses is outside the supported fitting API.

Public contract violations raise ValueError or RuntimeError as documented; incompatible checkpoint type raises TypeError. Conversion, malformed-file and filesystem exceptions propagate. Resource limits such as arbitrarily large requested networks are not exhaustively constrained by argument validation.

## Gradients, heuristics and constraints

`_accumulate_gradients` is the method used by train. `_grad_layers` always holds analytical gradients of E; `_heuristic_grad_layers` contains additive first-layer center-surrogate corrections; `_effective_grad_layers` combines them when enabled. These are private diagnostic structures, not a public cached-gradient API.

With legacy_heuristics=True (default), a first-layer small-exponential branch substitutes exp(-2*(x-C)**2) into a center update when exp(B*(x-C)**2)<0.001 and abs(hidden delta)>0.0001. The substitute-minus-exact contribution is stored separately. Width-relaxation branches use B*=0.9, freeze centers where specified and clear affected previous-gradient entries. The second-layer condition now uses exp(B*||h-C||**2), the same distance as its synapse, in both learning-rate branches.

With legacy_heuristics=False, these center-surrogate and small-exponential width/frozen-center rules are disabled. This does not disable correction_update_weight. Its historical rule resets a positive B to -0.005 in the first and second hidden layers; the supplied rule does not constrain output-layer B. Potential output overflow is detected, not silently constrained. This distinction is retained and documented rather than changing an optimization policy without validation.

`heuristic_events_` counts executions of center_surrogate, width_relaxation and positive_B_reset branches. Counts are not unique synapses and do not prove scientific benefit. Signs, normalized deltas and the corrected condition may change activations of historical thresholds: retaining policies does not recreate flawed historical trajectories.

## Momentum, batch and stopping

The original nonstandard momentum rule is preserved:

```math
\theta_{u+1}=\theta_u-\eta_e g_u-\mu g_{u-1}.
```

The previous value is the previous effective gradient accumulator, not a recursively accumulated optimizer velocity. Heuristic branches may reset selected previous-gradient components. The momentum term is not multiplied by the learning rate.

`model.batch=1` gives online updates. Any positive integer greater than one requests full-batch updates, not mini-batches of that size. Gradients are summed, not averaged over samples. Full-batch heuristic decisions use the last sample's cached activations/deltas at the update, preserving the supplied policy. Thus they are not exact derivatives of a batch loss. Sample order is the split order and is not reshuffled each epoch.

The epoch limit is inclusive: from epoch=0, max_epoch=5 allows six epochs, and max_epoch=0 performs one. `epoch` counts completed epochs. Starting at the second epoch, the stop counter increments when the absolute relative percentage change in epoch loss is strictly between zero and porcent_error_stop. It resets otherwise and stops after four consecutive qualifying changes. Small increases also count; an exact plateau does not. When previous loss is zero, change is zero if current loss is zero and infinity otherwise, preventing division by zero.

This is not validation-based early stopping, and no best-validation checkpoint is restored. `stop_reason_` is `epoch_limit` or `small_relative_change` after success. `error` records summed losses at each update, `error_epoch` their epoch sums, `error_val` summed scaled holdout losses every validation_frec completed epochs, and `date_time` epoch timestamps. Online loss sums span changing weights, not a frozen end-of-epoch model; they are not original-unit MSE.

## Persistence and exports

`save_model(path)` explicitly saves a successfully trained model; `load_model(path)` restores it and `load_VIHON` is an alias. Only load trusted pickle files: unpickling can execute code before type checks. Historical pickle compatibility and arbitrary future dependency compatibility are not guaranteed. Parent directories must exist; writes can overwrite existing files and are not transactional.

`save_weights(path=None)` returns a long-form CSV with columns `layer,neuron,input,parameter,component,value`. Vector parameters occupy three numeric rows, scalar parameters one row with an empty component. The default filename uses a portable timestamp and no model label. A 2–2–2–1 network exports 70 parameter rows. This is a diagnostic export, not a checkpoint/import format.

`load_data(path)` reads a rectangular CSV whose cells are literal three-component numeric vectors, without a header. Cells must be quoted by a proper CSV writer because vector text contains commas. Blank rows are skipped; empty, malformed and nonfinite datasets are rejected. Expressions are parsed with ast.literal_eval, never eval.

`predict_data(model,path,output_path=None)` returns a prediction matrix for all rows and writes a numeric prediction CSV only when output_path is explicit. It does not create an implicit file. `show_weights()` prints parameter coordinates and values.

## Validation limits

See validation/VALIDATION.md and the executed test log. Planned robustness checks are complete in the tested environment, not a proof that every possible failure is excluded. The evidence covers small synthetic networks, analytical gradients, selected optimizer branches, input contracts and export/lifecycle behavior. Historical performance benchmarks, arbitrary convergence, heuristic superiority, GPU execution and cross-platform numerical identity are not established.
