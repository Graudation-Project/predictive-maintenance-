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
        with open("../logs/training.txt",'a') as logs:

            x_train,x_test,y_train,y_test = train_test_split(self.X,self.y,test_size=.3,random_state=42)
            # Linear Regression
            lr = LinearRegression()
            self.logs.write("------------------------------------------\n")
            self.logs.write(f"Linear Regression Start Training {datetime.now()}")
            lr.fit(x_train,y_train)
            self.logs.write(f"Linear Regression Finished Training {datetime.now()}")
            ## Getting Time Latency Using One training Example
            start_time = time.time()
            lr.predict(x_train[0])
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
            self.logs.write("------------------------------------------\n")
            self.logs.write(f"Ridge Regression Start Training {datetime.now()}")
            param = {'alpha':[0.1,1,10,100]}
            grid_search = GridSearchCV(Ridge(),param_grid=param,scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(x_train,y_train)
            best_alpha = grid_search.best_params_["alpha"]

            ridge = Ridge(alpha=best_alpha)
            ridge.fit(x_train,y_train)
            self.logs.write(f"Ridge Regression Finished Training {datetime.now()}")
            test_pred = ridge.predict(x_test)
            start_time = time.time()
            ridge.predict(x_train[0])
            latency = time.time() - start_time
            self.logs.write(f"Calculating Metrices Starting... {datetime.now()}")
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
            self.logs.write(f"Calculating Metrices Finished.. {datetime.now()}")
            self.logs.write(f"Result Model... {datetime.now()}\n{self.metrics["Ridge Regression"]}")


            # Random Forest Regressor

            self.logs.write("------------------------------------------\n")
            self.logs.write(f"Random Forest Regressor Start Training {datetime.now()}")
            param_grid = {
                'n_estimators': [50, 100, 200],        
                'max_depth': [None, 10, 20, 30],      
                'min_samples_split': [2, 5, 10],      
                'min_samples_leaf': [1, 2, 4],       
                'max_features': ['auto', 'sqrt'],     
                'bootstrap': [True, False]
            }         

            grid_search = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                verbose=1
            )
            grid_search.fit(x_train, y_train)
            rfr = RandomForestRegressor(
                n_estimators=grid_search.best_params_["n_estimators"],
                max_depth=grid_search.best_params_["max_depth"],
                min_samples_split=grid_search.best_params_["min_samples_split"],
                min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                max_features=grid_search.best_params_["max_features"],
                bootstrap=grid_search.best_params_["bootstrap"]
            )
            self.logs.write(f"Random Forest Regressor Finished Training {datetime.now()}")
            test_pred = rfr.predict(x_test)
            start_time = time.time()
            rfr.predict(x_train[0])
            latency = time.time()-start_time
            self.logs.write(f"Calculating Metrices Starting... {datetime.now()}")
            self.models["Random Forest Regressor"] = rfr
            self.metrics["Random Forest Regressor"]["r2 score"] = r2_score(y_test,test_pred)
            self.metrics["Random Forest Regressor"]["mean absolute error"] = mean_absolute_error(y_test,test_pred)
            self.metrics["Random Forest Regressor"]["Cross Validation Score"] = cross_val_score(
                ridge,
                x_train,y_train,
                cv=5,
                scoring="accuracy"
                )
            self.metrics["Random Forest Regressor"]["Latency"] = latency
            self.logs.write(f"Calculating Metrices Finished.. {datetime.now()}")
            self.logs.write(f"Result Model... {datetime.now()}\n{self.metrics["Random Forest Regressor"]}")
            






    


    
