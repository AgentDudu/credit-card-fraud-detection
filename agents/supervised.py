import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

class SupervisedAgent:
    def __init__(self, model_type="logistic", retrain_interval=1000):
        self.model_type = model_type
        self.retrain_interval = retrain_interval
        
        if model_type == "logistic":
            self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=20, class_weight='balanced', n_jobs=1)
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(n_estimators=20, scale_pos_weight=100, eval_metric='logloss', n_jobs=1)
            
        self.memory_X = []
        self.memory_y = []
        self.is_trained = False
        self.steps = 0
        
    def get_action(self, context):
        if not self.is_trained:
            return 0
            
        x = context.reshape(1, -1)
        action = self.model.predict(x)[0]
        return int(action)
        
    def update(self, action, context, reward, info):
        self.steps += 1
        
        if info['result'] in ['TP', 'FN']:
            true_label = 1
        else:
            true_label = 0 
            
        self.memory_X.append(context)
        self.memory_y.append(true_label)
        
        if self.steps % self.retrain_interval == 0:
            X_train = np.array(self.memory_X)
            y_train = np.array(self.memory_y)
            
            if len(np.unique(y_train)) > 1:
                self.model.fit(X_train, y_train)
                self.is_trained = True