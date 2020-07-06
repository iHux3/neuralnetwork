custom neural network in php

- multiple input neurons
- multiple neurons in the hidden layer
- one output value

- the amount of neurons can be modified when creating an instance
  $nn = new NeuralNetwork(2,10); 
    @p0 number of input neurons (default 2)
    @p1 number of hidden neurons (default 2)

- to make neural network learn
  $nn->learn($targets,$inputs,$params);
    @p0 1d array containing values that the neural network should predict
    @p1 2d array containing values the neural network should learn from
    @p2 associative array, not required
  
- to predict
  $nn->predict([0,1]);

- an example of use is in file main.php
