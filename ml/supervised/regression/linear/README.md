##### Mathematics behind the weights initialization
A “gold solution” of the weights initialization on the internet, which states that “a good idea to choose initial weights of a neural network randomly from the range 

 <img src="https://latex.codecogs.com/svg.latex?\Large&space;[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}]" />
,where n is the number of hidden nodes in the input layer.



Under the assumption “the inputs are normalized to have a mean of 0 and variance (standard deviation^2) of 1”, the sum of the linear addition of all the outputs from the same layer before activation would have mean of 0 and variance of 1/3. At this range, the value of the derivative of the activation function would be reasonable. So our learning efficiency would be good.




For uncorrelated two randomly variables, 

<img src="https://latex.codecogs.com/svg.latex?&space;Var(X+Y) = Var(X) + Var(Y)" />

<img src="https://latex.codecogs.com/svg.latex?&space;Var(XY) = Var(X).Var(Y)" />

This is always true if we have more variables (see Variance Property on Wikipedia). The variance of uniform distribution
  <img src="https://latex.codecogs.com/svg.latex?\tiny&space;[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}]" /> is  <img src="https://latex.codecogs.com/svg.latex?\tiny&space;\frac{1}{3n}"/>.
 
 (see Uniform Distribution Property on Wikipedia). So the sum of the variances of all n nodes in the input layer would have a variance of 
   <img src="https://latex.codecogs.com/svg.latex?\tiny&space; \frac{1}{3n}\times{n}=\frac{1}{3}" />.
   