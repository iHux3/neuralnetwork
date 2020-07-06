<?php
  header('Content-type: text/plain');
  set_time_limit(60 * 60 * 24);
  require_once("./neuralnetwork.php");
  $nn = new NeuralNetwork(2,10);

  $inputs = [
    [1,1],
    [1,0],
    [0,1],
    [0,0],
  ];
  $targets = [0,1,1,0];

  $nn->learn($targets,$inputs);
  //echo "total loops: " . $nn->total_loops . PHP_EOL;
  echo "result: " . $nn->predict([1,1],1) . PHP_EOL;
?>
