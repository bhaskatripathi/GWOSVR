# GWO_RBF_SVM
This work uses "Grey Wolf Optimizer" to forecast the day-ahead prices of a financial asset (like Ethereum cryptocurrency). The Predictor variable is "US dollar exchange price" of Ethereum's blockchain. The Hybrid algorithm demonstrates the modelling of a complex case study of a chaotic dataset exhibiting properties like high-dimensionality, multimodality, non-uniformity and non-linearity. 

"The GWO algorithm mimics the leadership hierarchy and hunting mechanism of grey wolves in nature. Four types of grey wolves such as alpha, beta, delta, and omega are employed for simulating the leadership hierarchy. In addition, three main steps of hunting, searching for prey, encircling prey, and attacking prey, are implemented to perform optimization" : Seyedali Mirjalili (2020). Grey Wolf Optimizer (GWO) (https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo), MATLAB Central File Exchange. Retrieved May 28, 2020.

![GWO_RBF_SVM](https://in.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/44974/versions/9/screenshot.jpg)
* **Source:** - https://www.sciencedirect.com/science/article/pii/S0965997813001853

Advanced econometric models like ARCH,GARH,ARIMA etc fails to cater the aforementioned aspects of a noisy dataset. Therefore, using a stochastic model like GWO is much desired. A Support Vector Regressor(SVR) as an objective function with the intent of minimizing its fitness value is used to optimize the Cost (C) and Gamma(γ) hyper-parameters of SVR. Once the SVR is trained by GWO, it can be used to forecast the prices of Ethereum.

The Hybrid Methodology is then compared with other State-of-the-art benchmark optimization methodologies like Grid Search, PSO, Ant Colony Method etc. The results indicate a **superior performance** of this model over all other comparable models on evaluation measures of RMSE, MSE, MAPE, R-Square. 

You can star the project if it userful in your forecasting casestudies.
