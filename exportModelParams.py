import sklearn
import pickle
import pandas as pd
import numpy as np
import json

with open('bestPredictor.pickle', 'rb') as f:
    pipeline = pickle.load(f)
coefVars = ['AnkleAngleP','AnkleAngleR','AnkleAngleF', 'MTPAngleP','MTPAngleR','MTPAngleF','FootLength', 'FootWidth', 'Weight','Sex']
PCR_X = pd.read_csv('PCR_X.csv')
PCR_X = PCR_X[coefVars]

PCR_y = np.genfromtxt("PCR_y.csv", delimiter=',')

X_means = pipeline.steps[0][1].transformers_[0][1]._scaler.mean_
X_scales = pipeline.steps[0][1].transformers_[0][1]._scaler.scale_
X_lambdas = pipeline.steps[0][1].transformers_[0][1].lambdas_

y_means = pipeline.steps[1][1].transformer_._scaler.mean_
y_scales =pipeline.steps[1][1].transformer_._scaler.scale_
y_lambdas = pipeline.steps[1][1].transformer_.lambdas_

coefs = pipeline.steps[1][1].regressor_.coef_
intercepts = pipeline.steps[1][1].regressor_.intercept_

pipelineExports = {'X_means': X_means.tolist(),
                   'X_scales': X_scales.tolist(),
                   'X_lambdas': X_lambdas.tolist(),
                   'y_means': y_means.tolist(),
                   'y_scales': y_scales.tolist(),
                   'y_lambdas': y_lambdas.tolist(),
                   'coefs': coefs.tolist(),
                   'intercepts': intercepts.tolist()}
with open("dynafoot_Inputs.json", "w") as f:  
    json.dump(pipelineExports, f) 