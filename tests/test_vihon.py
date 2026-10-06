import ast
import csv
import copy
import io
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.model_selection import train_test_split
from vihon import VIHON,sigmoid,load_model,load_data,predict_data,_parameters,_value
from audit_gradients_vihon import audit,assign

def flat(layers):
    return np.array([_value(w,p,j) for _,_,_,w,p,j in _parameters(SimpleNamespace(layers=layers))])

def update_node():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'src/vihon.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VIHON')
    train=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    return next(n for n in ast.walk(train) if isinstance(n,ast.If) and ast.unparse(n.test)=='batch_count == self.batch')

def do_update(m):
    code=compile(ast.fix_missing_locations(ast.Module(body=[update_node()],type_ignores=[])),'<actual-update>','exec')
    exec(code,{'np':np,'self':m,'batch_count':1,'sum_error':0.,'sum_error_epoch':0.,'X_train':np.zeros((1,2,3)),'d':0,'Delta':[0.1],'delta1':[0.1,0.1],'delta':np.zeros((2,3))})

class VIHONRobustness(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(42).uniform(0,256,(12,2,3))
        self.y=(0.2+self.x[:,0,0]/512).reshape(-1,1)
    def model(self,**kw):
        args=dict(name='test',n_in=2,n_out=1,n_hidden=[2,2],learn_rate=1e-4,random_state=42,porcent_error_stop=0,validation_frec=1)
        args.update(kw); return VIHON(**args)
    def test_invalid_dimensions(self):
        for kw in [dict(n_in=0),dict(n_out=True),dict(n_hidden=[]),dict(n_hidden=[2]),dict(n_hidden=[2,0]),dict(n_hidden=[2,1.5]),dict(n_hidden=[[2],[2]])]:
            with self.subTest(kw=kw),self.assertRaises(ValueError): self.model(**kw)
    def test_invalid_learning_rates(self):
        for v in [None,True,0,-1,np.nan,np.inf,[],[0.1,0.2,3],[0.1,0.01,0],[0.1,0.01,1.5],np.array(0.01)]:
            with self.subTest(v=v),self.assertRaises(ValueError): self.model(learn_rate=v)
    def test_invalid_controls(self):
        for kw in [dict(momentum=-1),dict(momentum=np.nan),dict(test_size=0),dict(test_size=1),dict(test_size=True),dict(validation_frec=0),dict(porcent_error_stop=-1),dict(random_state=-1),dict(random_state=True),dict(random_state=2**32),dict(legacy_heuristics=1),dict(correction_update_weight=1),dict(cross_validation=True)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError): self.model(**kw)
    def test_bad_training_shapes(self):
        for x,y in [(self.x.reshape(12,6),self.y),(self.x,self.y[:,0]),(self.x,self.y[:-1]),(self.x[:1],self.y[:1]),(self.x[:,:,0],self.y),(self.x[:,0:1,:],self.y)]:
            with self.subTest(shape=x.shape),self.assertRaises(ValueError): self.model().train(x,y,0)
    def test_nonfinite_training_data(self):
        for x,y in [(np.full_like(self.x,np.nan),self.y),(self.x,np.full_like(self.y,np.inf))]:
            with self.assertRaises(ValueError): self.model().train(x,y,0)
    def test_epochs_batch(self):
        for e in [-1,True,1.5]:
            with self.assertRaises(ValueError): self.model().train(self.x,self.y,e)
        m=self.model(); m.batch=0
        with self.assertRaises(ValueError): m.train(self.x,self.y,0)
    def test_extreme_sigmoid(self):
        with np.errstate(all='raise'):
            np.testing.assert_array_equal(sigmoid([-1e300,-1000,0,1000,1e300]),[0,0,0.5,1,1])
    def test_nonfinite_sigmoid(self):
        with self.assertRaises(FloatingPointError): sigmoid(np.nan)
    def test_gaussian_underflow_allowed(self):
        out=self.model()._evaluate(np.full((2,3),1e100)); self.assertTrue(np.isfinite(out).all())
    def test_overflow_detected(self):
        m=self.model(); m.layers[0].weights[0][0].vB[0]=1000; m.layers[0].weights[0][0].vC[0]=0
        with self.assertRaises(FloatingPointError): m._evaluate(np.ones((2,3)))
    def test_nonfinite_parameter_detected(self):
        m=self.model(); m.layers[1].weights[0][0].B=np.nan
        with self.assertRaises(FloatingPointError): m._evaluate(np.zeros((2,3)))
    def test_forward_dimension_guard(self):
        with self.assertRaises(ValueError): self.model()._evaluate([1,2,3])
    def test_predict_lifecycle_and_shape(self):
        m=self.model()
        with self.assertRaises(RuntimeError): m.predict(self.x[0])
        m.train(self.x,self.y,0)
        for v in [np.ones(6),np.full((2,3),np.nan)]:
            with self.assertRaises(ValueError): m.predict(v)
    def test_retraining_rejected(self):
        m=self.model(); m.train(self.x,self.y,0)
        with self.assertRaises(RuntimeError): m.train(self.x,self.y,1)
    def test_failed_training_unusable(self):
        m=self.model(); m.layers[0].weights[0][0].vA[0]=np.nan
        with self.assertRaises(FloatingPointError): m.train(self.x,self.y,0)
        with self.assertRaises(RuntimeError): m.predict(self.x[0])
        with self.assertRaises(RuntimeError): m.train(self.x,self.y,0)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError): m.save_model(Path(td)/'m.pkl')
    def test_invalid_input_can_be_corrected_before_start(self):
        m=self.model()
        with self.assertRaises(ValueError): m.train(self.x,self.y[:,0],0)
        m.train(self.x,self.y,0); self.assertTrue(m._trained)
    def test_constant_data(self):
        m=self.model(); m.train(np.ones((10,2,3)),np.full((10,1),3.),1)
        self.assertTrue(np.isfinite(m.predict(np.ones((2,3)))).all())
    def test_seed_reproducibility(self):
        a=self.model(); b=self.model(); np.testing.assert_array_equal(flat(a.layers),flat(b.layers))
        a.train(self.x,self.y,2); b.train(self.x,self.y,2)
        np.testing.assert_array_equal(flat(a.layers),flat(b.layers)); np.testing.assert_array_equal(a.error_epoch,b.error_epoch)
        np.testing.assert_array_equal(a.train_indices_,b.train_indices_)
        np.testing.assert_array_equal(a.predict(self.x[0]),b.predict(self.x[0]))
    def test_global_rng(self):
        state=random.getstate(); self.model(); self.assertEqual(state,random.getstate())
    def test_other_seed(self):
        self.assertFalse(np.array_equal(flat(self.model().layers),flat(self.model(random_state=7).layers)))
    def test_input_domain_and_target_isolation(self):
        ti,vi=train_test_split(np.arange(12),test_size=0.2,random_state=42)
        x=self.x.copy(); y=self.y.copy(); x[vi]=1000; y[vi]=500
        a=self.model(); b=self.model(); a.train(self.x,self.y,2); b.train(x,y,2)
        np.testing.assert_array_equal(flat(a.layers),flat(b.layers)); np.testing.assert_array_equal(a.error_epoch,b.error_epoch)
        np.testing.assert_array_equal(b._target_scaler.data_max_,self.y[ti].max(axis=0))
        np.testing.assert_array_equal(b._input_scaler.transform([[0,128,256]]),[[-1,0,1]])
        self.assertFalse(set(ti)&set(vi))
    def test_single_vector_and_multioutput(self):
        m=self.model(n_in=1,n_out=2); y=np.column_stack((self.y[:,0],1-self.y[:,0])); m.train(self.x[:,:1,:],y,1)
        self.assertEqual(m.predict(self.x[0,:1,:]).shape,(2,))
    def test_full_batch(self):
        m=self.model(); m.batch=2; m.train(self.x,self.y,1)
        self.assertEqual(m.batch,9); self.assertEqual(len(m.error),2)
    def test_schedule(self):
        m=self.model(learn_rate=[0.01,0.001,3]); np.testing.assert_allclose(m.learn_rate,[0.01,0.0055,0.001])
        m.train(self.x,self.y,3); self.assertEqual(m.epoch,4)
    def test_epoch_limit(self):
        m=self.model(); m.train(self.x,self.y,5); self.assertEqual(m.epoch,6); self.assertEqual(m.stop_reason_,'epoch_limit')
    def test_zero_loss_stop_guard(self):
        m=self.model(porcent_error_stop=4,validation_frec=100)
        def fake(x,y): m._evaluate(x); return 0.
        m.get_error=fake; m.train(self.x,self.y,2); self.assertEqual(m.epoch,3)
    def test_four_qualifying_changes(self):
        m=self.model(porcent_error_stop=4,validation_frec=100); count=[0]
        def fake(x,y):
            m._evaluate(x); error=0.99**(count[0]//9); count[0]+=1; return error
        m.get_error=fake; m.train(self.x,self.y,20); self.assertEqual(m.epoch,5); self.assertEqual(m.stop_reason_,'small_relative_change')
    def test_momentum_actual_update(self):
        m=self.model(momentum=0.3,legacy_heuristics=False,correction_update_weight=False); initial=flat(m.layers)
        for _,_,_,w,p,j in _parameters(SimpleNamespace(layers=m._grad_layers)): assign(w,p,j,0.1)
        for _,_,_,w,p,j in _parameters(SimpleNamespace(layers=m._layers_momentum)): assign(w,p,j,0.2)
        do_update(m); np.testing.assert_allclose(flat(m.layers),initial-1e-4*0.1-0.3*0.2,rtol=1e-12,atol=1e-12)
    def test_center_surrogates_separate(self):
        x=np.array([[0.9,0.8,0.7],[-0.9,-0.8,-0.7]]); target=np.array([0.99]); a=self.model(legacy_heuristics=False); b=self.model(legacy_heuristics=True)
        for m in (a,b):
            for w in m.layers[0].weights.flat: w.vB=[-100,-100,-100]; w.vC=[0,0,0]
            m._evaluate(x); m._accumulate_gradients(x,target); m._prepare_effective_gradients()
        np.testing.assert_array_equal(flat(a._grad_layers),flat(b._grad_layers))
        self.assertEqual(a.heuristic_events_['center_surrogate'],0); self.assertGreater(b.heuristic_events_['center_surrogate'],0)
        self.assertFalse(np.array_equal(flat(b._grad_layers),flat(b._effective_grad_layers)))
    def test_distance_both_schedule_branches(self):
        for epoch in (0,1):
            m=self.model(); m.epoch=epoch; m.layers[0].out[:]=0.5
            w=m.layers[1].weights[0][0]; w.B=-100; w.vC=[0.5,0,0]
            m._grad_layers[1].weights[0][0].vC[:]=[0.1,0.2,0.3]
            do_update(m); self.assertEqual(w.B,-90); np.testing.assert_array_equal(w.vC,[0.5,0,0])
    def test_disabled_width_rule(self):
        m=self.model(legacy_heuristics=False); m.layers[0].out[:]=0.5
        w=m.layers[1].weights[0][0]; w.B=-100; w.vC=[0.5,0,0]; do_update(m)
        self.assertEqual(w.B,-100); self.assertEqual(m.heuristic_events_['width_relaxation'],0)
    def test_positive_B_policy(self):
        m=self.model(); m.layers[0].weights[0][0].vB[0]=0.1; m.layers[1].weights[0][0].B=0.1
        do_update(m)
        self.assertEqual(m.layers[0].weights[0][0].vB[0],-0.005); self.assertEqual(m.layers[1].weights[0][0].B,-0.005)
        self.assertEqual(m.heuristic_events_['positive_B_reset'],2)
    def test_gradient_audit(self):
        self.assertTrue(audit().passed.all())
    def test_no_implicit_files_roundtrip(self):
        m=self.model(); before=set(Path.cwd().iterdir()); m.train(self.x,self.y,1); self.assertEqual(before,set(Path.cwd().iterdir()))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.pkl'; m.save_model(p); r=load_model(p)
            np.testing.assert_array_equal(m.predict(self.x[0]),r.predict(self.x[0])); self.assertEqual(m.heuristic_events_,r.heuristic_events_)
    def test_untrained_save_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.pkl'
            with self.assertRaises(RuntimeError): self.model().save_model(p)
            with p.open('wb') as stream: pickle.dump({'not':'VIHON'},stream)
            with self.assertRaises(TypeError): load_model(p)
    def test_csv_full_roundtrip(self):
        m=self.model(); m.train(self.x,self.y,0)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.csv'
            with p.open('w',newline='') as stream: csv.writer(stream).writerows([[repr(list(v)) for v in row] for row in self.x[:2]])
            np.testing.assert_allclose(load_data(p),self.x[:2]); out=Path(td)/'output.csv'
            predictions=predict_data(m,p,out); self.assertEqual(predictions.shape,(2,1)); self.assertTrue(out.exists())
    def test_csv_unsafe_empty_malformed_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.csv'
            for text in ['', '__import__("os")','"[1,2]"','"[1,2,3]", "[4,5,6]"\n"[1,2,3]"\n']:
                p.write_text(text)
                with self.assertRaises((ValueError,SyntaxError)): load_data(p)
    def test_weight_export_scalar_components(self):
        m=self.model()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'weights.csv'; self.assertEqual(m.save_weights(p),p)
            with p.open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),70)
            np.testing.assert_allclose([float(row['value']) for row in rows],flat(m.layers))
            self.assertEqual({row['component'] for row in rows},{'','0','1','2'})
    def test_missing_parent_is_not_created(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError): self.model().save_weights(Path(td)/'missing'/'weights.csv')

if __name__=='__main__': unittest.main()
