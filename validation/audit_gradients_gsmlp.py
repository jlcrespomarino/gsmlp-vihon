import ast
from pathlib import Path
import numpy as np
import pandas as pd
from gsmlp import GSMLP

def audit():
    source=Path(__file__).resolve().parents[1]/'src'/'gsmlp.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GSMLP')
    train=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    loop=next(n for n in ast.walk(train) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='d')
    code=compile(ast.fix_missing_locations(ast.Module(body=loop.body[1:3],type_ignores=[])),'<actual-train-gradients>','exec')
    rows=[]
    for outputs in (1,2):
        for seed in (0,7,42):
            m=GSMLP('audit',2,outputs,[2],1e-4,random_state=seed)
            x=np.array([0.2,-0.4]); target=np.linspace(0.3,0.7,outputs)
            m._evaluate(x)
            exec(code,{'np':np,'self':m,'X_train':np.array([x]),'y_train':np.array([target]),'d':0})
            for li,lay in enumerate(m.layers):
                for ni,neuron in enumerate(lay.weights):
                    for ii,w in enumerate(neuron):
                        for p in 'ABC':
                            analytical=getattr(m._grad_layers[li].weights[ni][ii],p); v=getattr(w,p)
                            for step in (1e-5,1e-6):
                                setattr(w,p,v+step); plus=m.get_error(x,target)
                                setattr(w,p,v-step); minus=m.get_error(x,target)
                                setattr(w,p,v); numeric=(plus-minus)/(2*step)
                                rows.append(dict(n_out=outputs,seed=seed,layer=li,neuron=ni,input=ii,parameter=p,step=step,analytical=analytical,numerical=numeric,absolute_error=abs(analytical-numeric),passed=bool(np.isclose(analytical,numeric,rtol=1e-4,atol=1e-7))))
    return pd.DataFrame(rows)
if __name__=='__main__':
    result=audit(); result.to_csv(Path(__file__).with_name('gradient_check_gsmlp.csv'),index=False)
    assert result.passed.all(); print(result.groupby('n_out').passed.agg(['sum','count']))
