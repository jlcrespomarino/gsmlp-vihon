from pathlib import Path
import numpy as np
import pandas as pd
from vihon import VIHON,_parameters,_value

def assign(w,name,index,v):
    if index is None: setattr(w,name,v)
    else: getattr(w,name)[index]=v

def audit():
    rows=[]
    for mode in (False,True):
        for n_out in (1,2):
            for seed in (0,7,42):
                m=VIHON('audit',2,n_out,[2,2],1e-4,random_state=seed,legacy_heuristics=mode)
                x=np.array([[0.2,-0.4,0.3],[-0.1,0.5,-0.2]]); target=np.linspace(0.3,0.7,n_out)
                m._evaluate(x); m._accumulate_gradients(x,target)
                for li,ni,ii,w,name,index in _parameters(m):
                    g=_value(m._grad_layers[li].weights[ni][ii],name,index); initial=_value(w,name,index)
                    for step in (1e-5,1e-6):
                        assign(w,name,index,initial+step); plus=m.get_error(x,target)
                        assign(w,name,index,initial-step); minus=m.get_error(x,target)
                        assign(w,name,index,initial); numeric=(plus-minus)/(2*step)
                        rows.append(dict(legacy_heuristics=mode,n_out=n_out,seed=seed,layer=li,neuron=ni,input=ii,parameter=name,component=index,step=step,analytical=g,numerical=numeric,absolute_error=abs(g-numeric),passed=bool(np.isclose(g,numeric,rtol=1e-4,atol=1e-7))))
    return pd.DataFrame(rows)

if __name__=='__main__':
    result=audit(); result.to_csv(Path(__file__).with_name('gradient_check_vihon.csv'),index=False)
    assert result.passed.all(); print(result.groupby(['legacy_heuristics','layer']).passed.agg(['sum','count']))
