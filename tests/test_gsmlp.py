import copy
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from gsmlp import GSMLP, sigmoid, load_model, load_data, predict_data
from audit_gradients_gsmlp import audit

def parameters(m):
    return np.array([getattr(w,p) for lay in m.layers for neuron in lay.weights for w in neuron for p in 'ABC'])

def reference_gradients(m,x,target):
    output=m._evaluate(x); hidden=m.layers[0].out.copy()
    result=[]
    delta=(output-target)*output*(1-output)/len(target)
    for li,lay in enumerate(m.layers):
        layer_result=[]
        for j,neuron in enumerate(lay.weights):
            neuron_result=[]
            for i,w in enumerate(neuron):
                if li==1:
                    v=hidden[i]; common=delta[j]*v*np.exp(w.B*(v-w.C)**2)
                else:
                    v=x[i]
                    total=sum(delta[k]*ow.get_w(hidden[j])*(1+2*hidden[j]*ow.B*(hidden[j]-ow.C)) for k,ow in enumerate(m.layers[1].weights[:,j]))
                    common=v*hidden[j]*(1-hidden[j])*total*np.exp(w.B*(v-w.C)**2)
                neuron_result.append(np.array([common, common*w.A*(v-w.C)**2, -2*common*w.A*w.B*(v-w.C)]))
            layer_result.append(neuron_result)
        result.append(np.array(layer_result))
    return result

class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.x=np.linspace(-1,1,40).reshape(20,2)
        self.y=(0.5+0.2*self.x[:,0]).reshape(-1,1)
    def model(self,**kw):
        args=dict(name='test',n_in=2,n_out=1,n_hidden=[2],learn_rate=1e-4,random_state=42,porcent_error_stop=0,validation_frec=1)
        args.update(kw); return GSMLP(**args)
    def test_invalid_dimensions(self):
        for kw in [dict(n_in=0),dict(n_out=True),dict(n_hidden=[]),dict(n_hidden=[1.5]),dict(n_hidden=[0])]:
            with self.subTest(kw=kw),self.assertRaises(ValueError): self.model(**kw)
    def test_invalid_learning_rates(self):
        for v in [None,0,-1,True,np.nan,np.inf,[],[0.1,0.2,4],[0.1,0.01,0],[0.1,0.01,2.5]]:
            with self.subTest(v=v),self.assertRaises(ValueError): self.model(learn_rate=v)
    def test_invalid_controls(self):
        for kw in [dict(momentum=-1),dict(momentum=np.nan),dict(test_size=0),dict(test_size=1),dict(test_size=True),dict(validation_frec=0),dict(porcent_error_stop=-1),dict(random_state=-1),dict(random_state=True),dict(random_state=2**32),dict(correction_update_weight=1),dict(cross_validation=True)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError): self.model(**kw)
    def test_invalid_training_arrays(self):
        for x,y in [(self.x,self.y[:-1]),(self.x,self.y[:,0]),(self.x[:1],self.y[:1]),(self.x[:,0],self.y),(np.full_like(self.x,np.nan),self.y),(self.x,np.full_like(self.y,np.inf))]:
            with self.subTest(shape=np.shape(x)),self.assertRaises(ValueError): self.model().train(x,y,1)
    def test_invalid_epochs_batch(self):
        for v in [-1,True,1.5]:
            with self.subTest(v=v),self.assertRaises(ValueError): self.model().train(self.x,self.y,v)
        m=self.model(); m.batch=0
        with self.assertRaises(ValueError): m.train(self.x,self.y,1)
    def test_sigmoid_extremes(self):
        with np.errstate(all='raise'):
            # Underflow to zero is harmless for saturated sigmoid.
            with np.errstate(under='ignore'):
                out=sigmoid(np.array([-1e300,-1000,0,1000,1e300]))
        np.testing.assert_array_equal(out,[0,0,0.5,1,1])
    def test_nonfinite_sigmoid(self):
        with self.assertRaises(FloatingPointError): sigmoid(np.nan)
    def test_large_gaussian_underflow(self):
        m=self.model(); out=m._evaluate([1e100,-1e100]); self.assertTrue(np.isfinite(out).all())
    def test_overflow_detected(self):
        m=self.model(correction_update_weight=False); w=m.layers[0].weights[0][0]; w.B=1000; w.C=0
        with self.assertRaises(FloatingPointError): m._evaluate([1,1])
    def test_nonfinite_parameter_detected(self):
        m=self.model(); m.layers[0].weights[0][0].A=np.nan
        with self.assertRaises(FloatingPointError): m._evaluate([0.1,0.2])
    def test_input_prediction_guards(self):
        m=self.model()
        with self.assertRaises(RuntimeError): m.predict([0,0])
        m.train(self.x,self.y,0)
        for v in [[1,2,3],[np.inf,0],[[0,0]]]:
            with self.assertRaises(ValueError): m.predict(v)
    def test_constant_features_targets(self):
        m=self.model(); m.train(np.ones((10,2)),np.full((10,1),3.0),2)
        self.assertTrue(np.isfinite(m.predict([1,1])).all())
    def test_seed_reproducibility(self):
        a=self.model(); b=self.model(); np.testing.assert_array_equal(parameters(a),parameters(b))
        a.train(self.x,self.y,5); b.train(self.x,self.y,5)
        np.testing.assert_array_equal(parameters(a),parameters(b)); np.testing.assert_array_equal(a.error_epoch,b.error_epoch)
        np.testing.assert_array_equal(a.train_indices_,b.train_indices_)
        np.testing.assert_array_equal(a.predict([0.1,0.2]),b.predict([0.1,0.2]))
    def test_global_rng_untouched(self):
        state=random.getstate(); self.model(); self.assertEqual(state,random.getstate())
    def test_other_seed(self):
        self.assertFalse(np.array_equal(parameters(self.model()),parameters(self.model(random_state=7))))
    def test_holdout_isolation(self):
        ti,vi=train_test_split(np.arange(20),test_size=0.2,random_state=42)
        x=self.x.copy(); y=self.y.copy(); x[vi]=1000; y[vi]=500
        a=self.model(); b=self.model(); a.train(self.x,self.y,2); b.train(x,y,2)
        np.testing.assert_array_equal(parameters(a),parameters(b))
        np.testing.assert_array_equal(b._input_scaler.data_min_,self.x[ti].min(axis=0))
        np.testing.assert_array_equal(b._input_scaler.data_max_,self.x[ti].max(axis=0))
        np.testing.assert_array_equal(b._target_scaler.data_max_,self.y[ti].max(axis=0))
        self.assertFalse(set(ti)&set(vi))
    def test_epoch_limit(self):
        m=self.model(); m.train(self.x,self.y,5); self.assertEqual(m.epoch,6); self.assertEqual(m.stop_reason_,'epoch_limit')
    def test_retraining_rejected(self):
        m=self.model(); m.train(self.x,self.y,0)
        with self.assertRaises(RuntimeError): m.train(self.x,self.y,1)
    def test_zero_error_stop_guard(self):
        m=self.model(porcent_error_stop=4,validation_frec=100)
        def fake(x,y): m._evaluate(x); return 0.0
        m.get_error=fake; m.train(self.x,self.y,2); self.assertEqual(m.epoch,3)
        np.testing.assert_array_equal(m.error_epoch,[0,0,0])
    def test_four_changes_stop(self):
        m=self.model(porcent_error_stop=4,validation_frec=100); count=[0]
        train_count=16
        def fake(x,y):
            m._evaluate(x); value=0.99**(count[0]//train_count); count[0]+=1; return value
        m.get_error=fake; m.train(self.x,self.y,20)
        self.assertEqual(m.epoch,5); self.assertEqual(m.stop_reason_,'small_relative_change')
    def test_learning_rate_schedule(self):
        m=self.model(learn_rate=[0.1,0.01,3]); np.testing.assert_allclose(m.learn_rate,[0.1,0.055,0.01])
        m.train(self.x,self.y,3); self.assertEqual(m.epoch,4)
    def test_full_batch(self):
        m=self.model(); m.batch=2; m.train(self.x,self.y,1)
        self.assertEqual(m.batch,16); self.assertEqual(len(m.error),2)
    def test_momentum_matches_reference(self):
        m=self.model(momentum=0.2); ref=copy.deepcopy(m)
        ti,vi=train_test_split(np.arange(20),test_size=0.2,random_state=42)
        x=ref._input_scaler.fit_transform(self.x[ti]); y=ref._target_scaler.fit_transform(self.y[ti])
        previous=[np.zeros((2,2,3)),np.zeros((1,2,3))]
        for datum,target in zip(x,y):
            gradients=reference_gradients(ref,datum,target)
            for li,lay in enumerate(ref.layers):
                for ni,neuron in enumerate(lay.weights):
                    for ii,w in enumerate(neuron):
                        for pi,p in enumerate('ABC'):
                            setattr(w,p,getattr(w,p)-1e-4*gradients[li][ni,ii,pi]-0.2*previous[li][ni,ii,pi])
                        if w.B>0: w.B=-w.B
            previous=gradients
        m.train(self.x,self.y,0); np.testing.assert_allclose(parameters(m),parameters(ref),rtol=1e-12,atol=1e-12)
    def test_multihidden(self):
        m=self.model(n_hidden=[2,2]); self.assertEqual(m._evaluate([0,0]).shape,(1,))
        with self.assertRaises(NotImplementedError): m.train(self.x,self.y,1)
    def test_multioutput(self):
        m=self.model(n_out=2); m.train(self.x,np.column_stack((self.y[:,0],1-self.y[:,0])),2)
        self.assertEqual(m.predict([0.1,0.2]).shape,(2,))
    def test_persistence_and_no_implicit_files(self):
        m=self.model(); before=set(Path.cwd().iterdir()); m.train(self.x,self.y,1); self.assertEqual(before,set(Path.cwd().iterdir()))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'m.pkl'; m.save_model(path); loaded=load_model(path)
            np.testing.assert_array_equal(m.predict([0,0]),loaded.predict([0,0]))
            np.testing.assert_array_equal(m.validation_indices_,loaded.validation_indices_)
    def test_untrained_save_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError): self.model().save_model(Path(td)/'m.pkl')
    def test_csv_read_predict_export(self):
        m=self.model(); m.train(self.x,self.y,1)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.csv'; p.write_text('0.1,0.2\n0.3,0.4\n')
            self.assertEqual(load_data(p).shape,(2,2)); out=Path(td)/'result.csv'
            result=predict_data(m,p,out); self.assertEqual(result.shape,(2,1)); self.assertTrue(out.exists())
            weights=Path(td)/'weights.csv'; m.save_weights(weights); self.assertIn('layer,neuron,input,A,B,C',weights.read_text())
    def test_gradient_audit(self):
        self.assertTrue(audit().passed.all())

    def test_positive_B_reflection(self):
        args=dict(name='reflection',n_in=1,n_out=1,n_hidden=[1],learn_rate=10000,random_state=0,validation_frec=100,porcent_error_stop=0)
        free=GSMLP(**args,correction_update_weight=False); corrected=GSMLP(**args,correction_update_weight=True)
        free.train([[0],[1]],[[0],[1]],0); corrected.train([[0],[1]],[[0],[1]],0)
        free_b=np.array([w.B for lay in free.layers for row in lay.weights for w in row])
        fixed_b=np.array([w.B for lay in corrected.layers for row in lay.weights for w in row])
        self.assertTrue((free_b>0).any()); np.testing.assert_allclose(fixed_b,-np.abs(free_b))
    def test_failed_run_not_predictable_or_reusable(self):
        m=self.model(); m.layers[0].weights[0][0].A=np.nan
        with self.assertRaises(FloatingPointError): m.train(self.x,self.y,0)
        with self.assertRaises(RuntimeError): m.predict([0,0])
        with self.assertRaises(RuntimeError): m.train(self.x,self.y,0)

if __name__=='__main__': unittest.main()
