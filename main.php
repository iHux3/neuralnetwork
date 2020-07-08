<?php
  header('Content-type: text/plain');
  set_time_limit(60 * 60 * 24);
  require_once("./neuralnetwork.php");

  $inputs = [
    [1,1],
    [1,0],
    [0,1],
    [0,0]
  ];
  $targets = [1,1,1,0];

  $nn = new NeuralNetwork(2,2);
  $nn->learn($targets,$inputs);
  echo "result: " . $nn->predict([0,0]) . PHP_EOL;
?>
