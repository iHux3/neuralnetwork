<?php
  header('Content-type: text/plain');
  set_time_limit(60 * 60 * 24); //some calculations may take more time than default max time limit
  require_once("./neuralnetwork.php"); 
  $nn = new NeuralNetwork(2,2); //p0 - number of input neurons, p1 - number of hidden neurons

  $inputs = [
    [1,1],
    [1,0],
    [0,1],
    [0,0],
  ]; //an array of numbers (any numbers) that you want the neural network to learn from (must be 2d array)
  $targets = [0,1,1,0]; //an array of numbers (between 0 and 1) that you want the neural network to predict based on $inputs (must be an 1d array and must contain as many numbers as $inputs has rows)

  $nn->learn($targets,$inputs); //make neural network learn
  echo "result: " . $nn->predict([1,1],1) . PHP_EOL; //p0 must be an 1d array of numbers, p1 if true -> output will be 0 or 1, if false (default) -> output will be more accurate (between 0 and 1)
?>
