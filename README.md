# EE-399-A HW1

## Curve Fitting
Marcus Chen, 7 April 2023

Curve fitting is one of the most foundational concepts of machine learning and the studies of data sciences and artificial intelligence as a whole. In this project, we will observe the different aspects of the curve fitting: optimization based on the least square error, function modeling, and data selection for training. 

### Introduction and Overview:
Curve fitting is the process of constructing a curve, or mathematical function, that best summarizes or “fits” a series of discrete data points based on a presumed model. Because of its continuous properties, it can be used to both find patterns in a given data set and predict the values of data points in regions not specified by only the data; this property is why it is the most important foundational concept of data sciences, and machine learning in specific. All artificial intelligence technologies: ChatGPT, facial recognition, object detection, etc. are basically curve fits but involve high-dimensional data. 
```
X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
In this project, datasets X and Y are used to explore the different aspects of curve fitting. In the first experiment, the function model f, below, is used to determine the best curve fit coefficients based on an optimization done with the least square error. 

$f(x) = A cos(Bx) + Cx + D$

![399_8](https://user-images.githubusercontent.com/66970342/230534936-11b77a3e-5727-4196-922d-d5a39f803c40.png)


The four coefficient values A, B, C, and D are then treated as the control as two variables are sweeped and two variables are held constant in order to create a 2D loss (error) landscape, based on least-square error values for each combination of swept values. This is done for every combination of A, B, C, and D. 


In the next experiment, function modeling and training sets are tested. In one training set, the first 20 values of the data sets are used in order to predict the values of the remaining 11 values. In the second training set, the first 10 and last 10 values of the data sets are used to predict the values of the middle 11 values. These are done over a linear model, a parabolic model, and a 19th degree polynomial model.

### Theoretical Background:
#### Function Modeling
Modeling is the process of determining a function that will be used to predict the best curve fit of the data, where $F_\theta$ is the function, $X$ is the input, $Y$ is the output, and $\theta$ are the parameters. 

$Y = F_\theta(X)$

ex: $F_\theta(X) = AX + B,$
     $\theta \in {A, B}$

#### Least Square Error
The standard approach to regression analysis, or the analysis between the relationship between a curve and its data points. It is typically denoted by this formula:

$E = \sqrt{\frac{1}{n} \sum_{i=1}^n |f(x_i) - y_i|^2}$

The least squares regression model can also be used to find the coefficients of the modeled curve fit, by using the summation inside E. In the case of a linear function, f(x) = Ax + B, we can add the function to the formula and use the derivative to find the minima: 

$\varepsilon_2 = \sum_{i=1}^n |Ax_i + B - y_i|^2$

$\frac{\partial \varepsilon_2}{\partial A} = 0$

$\frac{\partial \varepsilon_2}{\partial B} = 0$

$A\Sigma x_i^2 + B \Sigma x_i = \Sigma x_i y_i$

$A\Sigma x_i + nB = \Sigma y_i$

#### Training and Testing
Another important aspect of the curve fitting process is training and testing data. The dataset provided can be divided into two sections: data that is used to create a function based on least squares regression, or training data, and data that is used to determine the accuracy of a model for prediction, testing data. It is important to make sure that the model function is able to accurately portray all data even if it is not added to the training set.

### Algorithm Interpretation and Development:
#### scipy.optimize.minimize
```
scipy.optimize.minimize(func, f0, args(X, Y), method='Nedler-Mead')
```
A function that inputs the linear regression model, arguments, an initial guess, and an optimization method, and returns a series of optimized coefficients in relation to the function model chosen. For this project, we are using the Nedler-Mead algorithm, which is often used on nonlinear models. 
#### numpy.polyfit
```
numpy.polyfit(X, Y, exp)
```
A function that inputs a series of data and a highest order exponent value and returns a series of coefficients for an optimized polynomial where the highest order is the imputed exponential value. 
#### matplotlib.pyplot.pcolor
```
matplotlib.pyplot.pcolor(C)
```
A function that creates a pseudocolor plot with a 2D array as an input. It creates a gradient between the lowest and highest values within the array.

### Computational Results:
#### Minima Sweep
In order to make the results more obvious, the least square errors are log scaled before being converted into a color map using pcolor. 


![399_1](https://user-images.githubusercontent.com/66970342/230534058-276337f5-844a-48fb-8f09-c9f92cbb025a.png)


The areas that are more blue indicate lower values, while the areas that are more yellow indicate higher values. The “stripes” of low values indicate that that specific coefficient has a more profound effect on the least square error at a given swept value. That is why in the situation where A and B are swept and B and D are swept, there are stripes of lower values along the B axis; and why in the situation where A and C, B and C, and C and D are swept, there are stripes of lower values along the C axis. In the situation where only B and C are swept, B has a more profound effect on the mean square error than C. 
Despite all that though, all the minima (or bluest) areas seem to be close to the predicted minimum through the optimize function. 

#### Data Selection

Trained with first 20 datapoints

![399_2](https://user-images.githubusercontent.com/66970342/230534331-6800d44d-495c-458b-81e0-9dcc6fd3e10e.png)
```
Training Data: 2.242749387090776
Testing Data: 3.363619366080294
```
![399_3](https://user-images.githubusercontent.com/66970342/230534350-a3685974-2a3a-49ce-85be-b4d0b9496c8f.png)
```
Training Data: 2.1255393483520155
Testing Data: 8.71366660302094
```
![399_4](https://user-images.githubusercontent.com/66970342/230534792-0d9073c5-674c-4ce2-b787-7592e48b2b38.png)
```
Training Data: 0.028351481277572182
Testing Data: 28626352734.19632
```

Trained with first 10 and last 10 datapoints

![399_5](https://user-images.githubusercontent.com/66970342/230534846-a337c07c-205e-40d6-a756-b000ba08c0d6.png)
```
Training Data: 1.8516699046029184
Testing Data: 2.73091076355018
```
![399_6](https://user-images.githubusercontent.com/66970342/230534850-c396e4dc-d742-4980-9235-9a333304919a.png)
```
Training Data: 1.8508364117779978
Testing Data: 2.7052339602955877
```
![399_7](https://user-images.githubusercontent.com/66970342/230534878-e5e96ba9-94c1-48e3-8eeb-f568263e2bd4.png)
```
Training Data: 532.5689882067522
Testing Data: 463.31672158927245
```


With all these comparisons, it is shown that the models that use training data from each end did an overall better job at predicting the values of the testing data than the model that only used the first 20 points to train. Moreover, the 19th degree polynomial model was extremely terrible with being able to predict the values of the testing data; while the latter training set did 55 million times better, it still had an error of over 400: 200 times the average error for the comparatively simpler linear and parabolic models. 

### Summary and Conclusions:
Through these exercises, we were able to analyze many aspects of curve fits. In the experiment with the swept data, changes to C affected the loss way more profoundly than some other coefficients like A and D, and the minima ultimately did agree with the predicted optimized values. In the experiment with selecting training data and functional model, training data that was more spread out was much more accurate in being able to predict the testing data values. The more complex the model was, the better it was for being able to encapsulate the training data, but at the 19th polynomial, it became extremely inaccurate to use it to predict other data. 

That phenomenon is commonly known as overfitting, or fitting the data too close to the training. Ultimately, many issues that we currently face with machine learning, such as racial bias in facial recognition or driving incompetency with self-driving vehicles, are because we often choose models that overfit to specific training data. Just like with the simple curve fitting with 1-dimensional lines, it is important to create a function that encapsulates the current data while also being able to be used on data beyond the training. 

### References 
* https://time.com/5520558/artificial-intelligence-racial-gender-bias/
* https://www.theguardian.com/technology/2022/aug/09/tesla-self-driving-technology-safety-children

