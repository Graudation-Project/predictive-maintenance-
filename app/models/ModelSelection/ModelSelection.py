from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
class ModelSelection:
    def __init__(self,data):
        self.models = {
            "Linear Regression":None,
            "Ridge":None,
            "Random Forest Regressor":None,
            "Deep Learning":None
        }
        self.metrics = {
            "Linear Regression":{},
            "Ridge":{},
            "Random Forest Regressor":{},
            "Deep Learning":{}
        }
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]
        np.random.seed(0)


    def training(self):
        with open("../logs/ModelLifeCycle.txt",'a') as logs:

            x_train,x_test,y_train,y_test = train_test_split(self.X,self.y,test_size=.3,random_state=42)
            # Linear Regression
            lr = LinearRegression()
            self.logs.write("------------------------------------------\n")
            self.logs.write(f"Linear Regression Start Training {datetime.now()}")
            lr.fit(x_train,y_train)
            self.logs.write(f"Linear Regression Finished Training {datetime.now()}")
            ## Getting Time Latency Using One training Example
            start_time = time.time()
            test = lr.predict(x_train[0])
            latency = time.time()-start_time
            test_pred = lr.predict(x_test)
            self.logs.write(f"Calculating Metrices Starting... {datetime.now()}")
            self.models["Linear Regression"] = lr
            self.metrics["Linear Regression"]["r2 score"] = r2_score(y_test,test_pred)
            self.metrics["Linear Regression"]["mean absolute error"] = mean_absolute_error(y_test,test_pred)
            self.metrics["Linear Regression"]["Cross Validation Score"] = cross_val_score(
                lr,
                x_train,y_train,
                cv=5,
                scoring="accuracy"
                )
            self.metrics["Linear Regression"]["Latency"] = latency
            self.logs.write(f"Calculating Metrices Finished.. {datetime.now()}")
            self.logs.write(f"Result Model... {datetime.now()}\n{self.metrics["Linear Regression"]}")
            # Ridge Regression
            ridge = Ridge()
            param = {'alpha':[0.1,1,10,100]}
            grid_search = GridSearchCV(ridge,param_grid=param,scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(x_train,y_train)
            best_alpha = grid_search.best_params_["alpha"]

            ridge = Ridge(alpha=best_alpha)
            ridge.fit(x_train,y_train)
            test_pred = ridge.predict(x_test)
            start_time = time.time()
            test = ridge.predict(x_train[0])
            latency = time.time() - start_time

            self.models["Ridge"] = ridge
            self.metrics["Ridge"]["r2 score"] = r2_score(y_test,test_pred)
            self.metrics["Ridge"]["mean absolute error"] = mean_absolute_error(y_test,test_pred)
            self.metrics["Ridge"]["Cross Validation Score"] = cross_val_score(
                ridge,
                x_train,y_train,
                cv=5,
                scoring="accuracy"
                )
            self.metrics["Ridge"]["Latency"] = latency





    


    
