
**Notebook Outline:**  
  
- [Setup with libraries](#Set-up-Cells)
- [Clearwater Landslides Dataset](#Clearwater-Landslides-Dataset)
- [Univariate example](#Univariate-example)
    - [Bandwidth check](#Bandwidth-check)
    - [Parameter check](#Parameter-check)
- [Multivariate example](#Multivariate-example)
    - [Bandwidths check](#Bandwidths-check)
    - [AIC, AICc, BIC check](#AIC,-AICc,-BIC-check)
- [Global model check](#Global-model-check)

### Set up Cells


```python
import sys
sys.path.append("C:/Users/msachde1/Downloads/Research/Development/mgwr")
```


```python
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from mgwr.gwr import GWR
from spglm.family import Gaussian, Binomial, Poisson
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
import multiprocessing as mp
pool = mp.Pool()
from scipy import linalg
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse import linalg as spla
from spreg.utils import spdot, spmultiply
from scipy import special
import libpysal as ps
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import copy
from collections import namedtuple
import spglm
```

### Clearwater Landslides Dataset

#### Clearwater data - downloaded from link: https://sgsup.asu.edu/sparc/multiscale-gwr


```python
data_p = pd.read_csv("C:/Users/msachde1/Downloads/logistic_mgwr_data/landslides.csv") 
```


```python
data_p.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UserID</th>
      <th>X</th>
      <th>Y</th>
      <th>Elev</th>
      <th>Slope</th>
      <th>SinAspct</th>
      <th>CosAspct</th>
      <th>AbsSouth</th>
      <th>Landslid</th>
      <th>DistStrm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>616168.5625</td>
      <td>5201076.5</td>
      <td>1450.475</td>
      <td>27.44172</td>
      <td>0.409126</td>
      <td>-0.912478</td>
      <td>24.1499</td>
      <td>1</td>
      <td>8.506</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>624923.8125</td>
      <td>5201008.5</td>
      <td>1567.476</td>
      <td>21.88343</td>
      <td>-0.919245</td>
      <td>-0.393685</td>
      <td>66.8160</td>
      <td>1</td>
      <td>15.561</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>615672.0000</td>
      <td>5199187.5</td>
      <td>1515.065</td>
      <td>38.81030</td>
      <td>-0.535024</td>
      <td>-0.844837</td>
      <td>32.3455</td>
      <td>1</td>
      <td>41.238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>615209.3125</td>
      <td>5199112.0</td>
      <td>1459.827</td>
      <td>26.71631</td>
      <td>-0.828548</td>
      <td>-0.559918</td>
      <td>55.9499</td>
      <td>1</td>
      <td>17.539</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>616354.6875</td>
      <td>5198945.5</td>
      <td>1379.442</td>
      <td>27.55271</td>
      <td>-0.872281</td>
      <td>-0.489005</td>
      <td>60.7248</td>
      <td>1</td>
      <td>35.023</td>
    </tr>
  </tbody>
</table>
</div>



### Univariate example

#### GWR Binomial model with independent variable, x = slope


```python
coords = list(zip(data_p['X'],data_p['Y']))
y = np.array(data_p['Landslid']).reshape((-1,1)) 
elev = np.array(data_p['Elev']).reshape((-1,1))
slope = np.array(data_p['Slope']).reshape((-1,1))
SinAspct = np.array(data_p['SinAspct']).reshape(-1,1)
CosAspct = np.array(data_p['CosAspct']).reshape(-1,1)
X = np.hstack([elev,slope,SinAspct,CosAspct])
x = CosAspct

X_std = (X-X.mean(axis=0))/X.std(axis=0)
x_std = (x-x.mean(axis=0))/x.std(axis=0)
y_std = (y-y.mean(axis=0))/y.std(axis=0)
```


```python
bw=Sel_BW(coords,y,x_std,family=Binomial(),constant=False).search()
gwr_mod=GWR(coords,y,x_std,bw=bw,family=Binomial(),constant=False).fit()
bw
```




    108.0



##### Running the function with family = Binomial()

#### Bandwidths check


```python
selector = Sel_BW(coords,y,x_std,family=Binomial(),multi=True,constant=False)
selector.search(verbose=True)
```

    Current iteration: 1 ,SOC: 0.0752521
    Bandwidths: 108.0
    Current iteration: 2 ,SOC: 0.0213201
    Bandwidths: 184.0
    Current iteration: 3 ,SOC: 5.8e-05
    Bandwidths: 184.0
    Current iteration: 4 ,SOC: 1e-06
    Bandwidths: 184.0
    




    array([184.])




```python
mgwr_mod = MGWR(coords, y,x_std,selector,family=Binomial(),constant=False).fit()
```


    HBox(children=(IntProgress(value=0, description='Inference', max=1), HTML(value='')))


    
    

#### Parameter check


```python
mgwr_mod.bic
```




    325.23949237389036




```python
gwr_mod.bic
```




    338.19722049287054




```python
mgwr_mod.predy-gwr_mod.predy
```




    array([[-6.86421110e-02],
           [-9.02377310e-03],
           [-6.25927989e-02],
           [-3.46325349e-02],
           [-2.46608964e-02],
           [-7.92934492e-02],
           [-7.42296482e-03],
           [-1.16179902e-02],
           [-8.55163917e-03],
           [ 6.69586863e-03],
           [-1.34781343e-02],
           [-4.69002263e-02],
           [-4.27190305e-02],
           [-1.96234952e-03],
           [-3.45017324e-02],
           [ 4.78018268e-03],
           [ 2.35625148e-02],
           [ 4.59257586e-03],
           [ 2.67844779e-02],
           [ 4.75083942e-03],
           [-5.13130035e-02],
           [-7.92681519e-02],
           [ 7.52785701e-02],
           [ 7.45723922e-02],
           [-5.19817611e-02],
           [-6.22604390e-02],
           [-1.73378137e-02],
           [-4.88622106e-02],
           [ 9.15948348e-03],
           [ 1.84910322e-02],
           [ 3.15425672e-03],
           [-6.48930429e-03],
           [-1.93762371e-03],
           [-4.34061793e-02],
           [-5.28568280e-02],
           [ 1.74571605e-02],
           [-4.73594024e-03],
           [-2.25230562e-03],
           [-1.10529379e-02],
           [-2.61542043e-02],
           [-5.47613290e-02],
           [-8.56136675e-02],
           [ 6.11606014e-02],
           [-7.96316019e-02],
           [ 1.06137092e-01],
           [-8.45634873e-02],
           [-5.16056559e-02],
           [-9.39907188e-02],
           [ 5.08910015e-02],
           [-1.11319637e-01],
           [-2.80789755e-02],
           [ 8.55069680e-02],
           [-4.68078891e-02],
           [-3.33390431e-03],
           [-2.16235723e-02],
           [ 2.49215314e-02],
           [ 5.63334013e-03],
           [ 4.03752737e-02],
           [-1.43170358e-01],
           [-1.59890083e-01],
           [-7.42351180e-02],
           [ 6.04559033e-02],
           [ 3.33959925e-02],
           [-8.40832151e-03],
           [-2.65428632e-02],
           [-8.67898410e-02],
           [-8.66556038e-02],
           [-6.84698662e-02],
           [-5.61187590e-02],
           [-5.67881208e-02],
           [-6.63001614e-04],
           [-1.06074298e-03],
           [ 3.15268253e-02],
           [-4.13201319e-02],
           [-6.20969024e-02],
           [-6.25418387e-02],
           [-6.43731665e-02],
           [ 5.05707341e-02],
           [-1.81040565e-02],
           [ 3.13435449e-02],
           [-1.35797422e-02],
           [-6.46775147e-02],
           [-1.53156899e-01],
           [ 8.11104441e-03],
           [ 6.41924414e-03],
           [ 6.74537680e-03],
           [-2.59694029e-02],
           [-6.08872689e-02],
           [-5.76580369e-02],
           [ 2.09339187e-02],
           [-1.84105049e-03],
           [ 3.22956968e-03],
           [ 7.27477155e-02],
           [-2.81118267e-02],
           [-3.45626639e-02],
           [-3.58294089e-02],
           [-6.09426122e-02],
           [ 5.17859563e-02],
           [-7.14232795e-02],
           [ 5.29332087e-02],
           [-7.58861089e-03],
           [-5.86114846e-03],
           [-6.87044228e-02],
           [ 1.28916049e-02],
           [-6.67529531e-02],
           [-6.69324628e-03],
           [ 9.56438588e-03],
           [-3.75737563e-03],
           [-2.64118699e-03],
           [ 1.35663378e-01],
           [ 1.97104440e-02],
           [-1.59758799e-01],
           [-4.44888861e-02],
           [ 4.20888604e-03],
           [-3.99928268e-02],
           [-4.68677717e-02],
           [-1.84088221e-01],
           [-1.08771012e-01],
           [-1.03004376e-01],
           [-7.20510889e-02],
           [-6.90533144e-02],
           [-7.37314396e-02],
           [ 7.50691754e-02],
           [ 3.53817571e-02],
           [ 4.30473649e-02],
           [ 1.94594454e-02],
           [ 6.96279182e-02],
           [-1.47661459e-02],
           [-2.55449197e-02],
           [-2.75477053e-02],
           [-3.53445515e-02],
           [ 4.28998056e-03],
           [-2.97887303e-03],
           [ 3.34820966e-02],
           [ 1.53836989e-02],
           [ 2.98504386e-02],
           [-5.44861948e-02],
           [-9.85317074e-02],
           [ 1.90976659e-01],
           [ 2.64677520e-02],
           [ 1.40322828e-01],
           [ 2.50343699e-01],
           [-9.26960534e-03],
           [ 3.09008565e-02],
           [ 5.59670178e-02],
           [ 1.02006130e-01],
           [-2.35602286e-02],
           [ 1.14163733e-01],
           [ 3.47931823e-02],
           [-1.11183101e-03],
           [ 1.39762958e-01],
           [-8.04828275e-02],
           [-3.39115588e-02],
           [-1.71467752e-02],
           [ 5.72292660e-04],
           [-2.16044180e-04],
           [ 1.28949115e-01],
           [ 1.26023648e-01],
           [ 8.84030834e-02],
           [ 2.99680764e-02],
           [ 5.56646040e-02],
           [-5.12025553e-02],
           [ 9.47313130e-03],
           [-5.36473598e-02],
           [-7.29087711e-02],
           [-6.92127110e-02],
           [-8.95235041e-03],
           [-1.21073353e-02],
           [ 1.04199536e-01],
           [-3.44823874e-02],
           [-5.67763767e-02],
           [ 1.13578616e-01],
           [ 4.52178521e-02],
           [ 1.24571646e-01],
           [ 8.28563723e-02],
           [ 2.42844475e-03],
           [-7.96453685e-02],
           [ 2.40809343e-02],
           [-1.55982333e-02],
           [ 7.46162805e-02],
           [ 1.01380277e-01],
           [ 1.20509132e-01],
           [ 1.35523796e-01],
           [-3.22828125e-02],
           [-1.37241610e-02],
           [-5.97213805e-02],
           [ 1.10999448e-01],
           [-3.62631169e-02],
           [ 1.84431180e-02],
           [ 1.01931343e-01],
           [-6.84139926e-02],
           [-3.45131451e-02],
           [-9.75344201e-03],
           [ 4.80032525e-02],
           [ 2.34601191e-02],
           [ 3.89468591e-02],
           [ 1.65250383e-01],
           [-2.02724746e-02],
           [ 1.47208967e-01],
           [ 5.84571936e-02],
           [ 2.42283823e-02],
           [ 2.42638484e-02],
           [-6.78773201e-03],
           [-1.20966266e-01],
           [-1.62437523e-02],
           [ 8.47043108e-02],
           [-9.80263831e-02],
           [-7.21865204e-02],
           [-6.04470168e-02],
           [-4.94343083e-03],
           [-7.30471221e-03],
           [ 1.01563662e-01],
           [-3.45435004e-02],
           [ 7.28656005e-02],
           [-2.87797340e-02],
           [ 1.16621688e-01],
           [ 3.04019792e-02],
           [-7.93027975e-02],
           [ 3.86078732e-03],
           [-5.30570863e-02],
           [-5.48487906e-02],
           [ 1.70763200e-01],
           [-4.14052105e-02],
           [ 1.31774383e-02],
           [-3.06852068e-02],
           [-8.35451055e-02],
           [ 2.58827225e-02],
           [-3.86634974e-02],
           [ 4.27430398e-02],
           [-2.89324464e-02],
           [ 1.76638263e-01],
           [ 1.03211895e-01],
           [ 6.55672059e-02],
           [-8.81854679e-02],
           [ 8.49001037e-02],
           [-8.95180777e-02],
           [ 1.21071638e-01],
           [ 1.27907248e-02],
           [-3.23289914e-03]])



### Multivariate example


```python
bw=Sel_BW(coords,y,X_std,family=Binomial(),constant=True).search()
gwr_mod=GWR(coords,y,X_std,bw=bw,family=Binomial(),constant=True).fit()
bw
```




    121.0



#### Bandwidth check


```python
selector = Sel_BW(coords,y,X_std,family=Binomial(),multi=True,constant=True)
selector.search(verbose=True)
```

    Current iteration: 1 ,SOC: 0.116124
    Bandwidths: 43.0, 62.0, 191.0, 100.0, 108.0
    Current iteration: 2 ,SOC: 0.0266811
    Bandwidths: 43.0, 106.0, 210.0, 100.0, 184.0
    Current iteration: 3 ,SOC: 0.0008147
    Bandwidths: 43.0, 106.0, 210.0, 100.0, 184.0
    Current iteration: 4 ,SOC: 5.28e-05
    Bandwidths: 43.0, 106.0, 210.0, 100.0, 184.0
    Current iteration: 5 ,SOC: 5.3e-06
    Bandwidths: 43.0, 106.0, 210.0, 100.0, 184.0
    




    array([ 43., 106., 210., 100., 184.])




```python
mgwr_mod = MGWR(coords, y,X_std,selector,family=Binomial(),constant=True).fit()
```


    HBox(children=(IntProgress(value=0, description='Inference', max=1), HTML(value='')))


    
    

#### AIC, AICc, BIC check


```python
gwr_mod.aicc, mgwr_mod.aicc
```




    (264.9819711678866, 251.85376815296377)



### Global model check


```python
selector=Sel_BW(coords,y,X_std,multi=True,family=Binomial(),constant=True)
selector.search(verbose=True,multi_bw_min=[239,239,239,239,239], multi_bw_max=[239,239,239,239,239])
```

    Current iteration: 1 ,SOC: 0.6120513
    Bandwidths: 239.0, 239.0, 239.0, 239.0, 239.0
    Current iteration: 2 ,SOC: 0.0594775
    Bandwidths: 239.0, 239.0, 239.0, 239.0, 239.0
    Current iteration: 3 ,SOC: 0.0025897
    Bandwidths: 239.0, 239.0, 239.0, 239.0, 239.0
    Current iteration: 4 ,SOC: 0.0001289
    Bandwidths: 239.0, 239.0, 239.0, 239.0, 239.0
    Current iteration: 5 ,SOC: 1.17e-05
    Bandwidths: 239.0, 239.0, 239.0, 239.0, 239.0
    Current iteration: 6 ,SOC: 1.2e-06
    Bandwidths: 239.0, 239.0, 239.0, 239.0, 239.0
    




    array([239., 239., 239., 239., 239.])




```python
mgwr_mod = MGWR(coords, y,X_std,selector,family=Binomial(),constant=True).fit()
```


    HBox(children=(IntProgress(value=0, description='Inference', max=1), HTML(value='')))


    
    


```python
gwr_mod.summary()
```

    ===========================================================================
    Model type                                                         Binomial
    Number of observations:                                                 239
    Number of covariates:                                                     5
    
    Global Regression Results
    ---------------------------------------------------------------------------
    Deviance:                                                           266.246
    Log-likelihood:                                                    -133.123
    AIC:                                                                276.246
    AICc:                                                               276.504
    BIC:                                                              -1015.246
    Percent deviance explained:                                           0.182
    Adj. percent deviance explained:                                      0.168
    
    Variable                              Est.         SE  t(Est/SE)    p-value
    ------------------------------- ---------- ---------- ---------- ----------
    X0                                   0.389      0.150      2.591      0.010
    X1                                  -0.784      0.166     -4.715      0.000
    X2                                   0.654      0.168      3.881      0.000
    X3                                   0.039      0.149      0.264      0.792
    X4                                  -0.371      0.156     -2.381      0.017
    
    Geographically Weighted Regression (GWR) Results
    ---------------------------------------------------------------------------
    Spatial kernel:                                           Adaptive bisquare
    Bandwidth used:                                                     121.000
    
    Diagnostic information
    ---------------------------------------------------------------------------
    Effective number of parameters (trace(S)):                           23.263
    Degree of freedom (n - trace(S)):                                   215.737
    Log-likelihood:                                                    -106.599
    AIC:                                                                259.725
    AICc:                                                               264.982
    BIC:                                                                340.598
    Percent deviance explained:                                         0.345
    Adjusted percent deviance explained:                                0.274
    Adj. alpha (95%):                                                     0.011
    Adj. critical t value (95%):                                          2.571
    
    Summary Statistics For GWR Parameter Estimates
    ---------------------------------------------------------------------------
    Variable                   Mean        STD        Min     Median        Max
    -------------------- ---------- ---------- ---------- ---------- ----------
    X0                        0.459      0.360     -0.360      0.436      1.232
    X1                       -0.824      0.479     -2.128     -0.729     -0.095
    X2                        0.567      0.390     -0.030      0.600      1.328
    X3                        0.103      0.270     -0.473      0.183      0.565
    X4                       -0.331      0.247     -1.118     -0.287      0.096
    ===========================================================================
    
    


```python
np.mean(mgwr_mod.params,axis=0)
```




    array([ 0.19936242, -0.3251776 ,  0.32069312,  0.04295657, -0.20408904])




```python
mgwr_mod.bic, gwr_mod.bic
```




    (303.9521120546862, 340.5982180538755)


