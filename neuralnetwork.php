<?php
	class NeuralNetwork{
		private $hidden_layer;
		private $output_layer;
		private $il;
		private $hl;
		private $learning_rate;
		private $inputs;
		private $targets;
		private $threshold;
		public $total_loops = 0;

		public function __construct($input_layers = 2,$hidden_layers = 2){
			if($input_layers < 1) $input_layers = 1;
			if($hidden_layers < 1) $hidden_layers = 1;
			$this->il = round($input_layers);
			$this->hl = round($hidden_layers);

			$output_weights = [];
			for($i = 0; $i < $hidden_layers; $i++){
				$hidden_weights = [];
				for($j = 0; $j < $input_layers; $j++){
					$hidden_weights[$j] = $this->random_weight();
				}
				$this->hidden_layer[$i] = ["weights" => $hidden_weights, "bias" => $this->random_weight()];
				$output_weights[$i] = $this->random_weight();
			}
			$this->output_layer = [
				["weights" => $output_weights, "bias" => $this->random_weight()]
			];
		}

		private function random_weight(){
			return rand(1,99) / 100;
		}

		private function sigmoid($f){
			return 1 / (1 + pow(M_E,-$f));
		}

		private function forward($input,$type,$id){
			$layer = $type == 0 ? $this->hidden_layer : $this->output_layer;
			$result = 0;
			for($i = 0; $i < count($input); $i++){
				$result += $input[$i] * $layer[$id]["weights"][$i];
			}
			$result += $this->hidden_layer[$id]["bias"];

			return self::sigmoid($result);
		}

		private function backward($o,$target,$h,$input){
			$pre = ($o - $target) * ($o * (1 - $o));

			for($i = 0; $i < count($this->output_layer[0]["weights"]); $i++){
				$this->output_layer[0]["weights"][$i] = $this->output_layer[0]["weights"][$i] - $this->learning_rate * ($pre * $h[$i]);
			}
			for($i = 0; $i < count($this->hidden_layer); $i++){
				for($j = 0; $j < count($this->hidden_layer[$i]["weights"]); $j++){
					$this->hidden_layer[$i]["weights"][$j] = $this->hidden_layer[$i]["weights"][$j] - $this->learning_rate * ($pre * $this->output_layer[0]["weights"][$i] * ($h[$i] * (1 - $h[$i])) * $input[$j]);
				}
			}
		}

		private function error($target,$o){
			return $this->learning_rate * pow($target - $o,2);
		}

		public function learn($targets,$inputs,$errMin = 0.01,$max_loops = 10000,$learning_rate = 0.1){
			try{
				if(!is_array($targets)) throw new Exception('$targets must be an array');
				foreach($targets as $target){
					if(is_array($target)) throw new Exception('$targets must have exactly one column');
					if(!is_numeric($target)) throw new Exception('$targets non-numeric value detected');
					if($target < 0 || $target > 1) throw new Exception('$targets values must be between 0 and 1' . PHP_EOL);
				}
				if(count($targets) != count($inputs)) throw new Exception('$targets must contain the same number of rows as $inputs');

				if(!is_array($inputs)) throw new Exception('$inputs must be an array');
				foreach($inputs as $input){
					if(!is_array($input)) throw new Exception('$inputs must be a 2d array');
					if(count($input) != $this->il) throw new Exception('$inputs must contain ' . $this->il . ' columns');
					foreach($input as $i){
						if(!is_numeric($i)) throw new Exception('$inputs non-numeric value detected');
					}
				}
				if($errMin <= 0) throw new Exception('$errMin must be greater than zero');
				if($learning_rate <= 0) throw new Exception('$learning_rate must be greater than zero');

				$this->learning_rate = $learning_rate;
				$this->targets = $targets;
				$this->inputs = $inputs;

				$this->total_loops = 0;
				do{
					$errSum = 0;
					for($j = 0; $j < count($targets); $j++){
						$target = $targets[$j];
						$input = $inputs[$j];

						$h = [];
						for($k = 0; $k < count($this->hidden_layer); $k++) $h[$k] = $this->forward($input,0,$k);
						$o = $this->forward($h,1,0);
						$err = $this->error($target,$o);
						$errSum += $err;
						$this->backward($o,$target,$h,$input);
					}
					$this->total_loops++;
				}while($errSum > $errMin && $this->total_loops < $max_loops);
			}catch(Exception $e){
				echo "Error: " . $e->getMessage();
			}

		}

		public function predict($input){
			try{
				if(!is_array($input)) throw new Exception('$input must be an array');
				if(count($input) != $this->il) throw new Exception('$input must contain ' . $this->il . ' values');
				foreach($input as $i){
					if(!is_numeric($i)) throw new Exception('$input non-numeric value detected');
				}
				$h = [];
				for($i = 0; $i < count($this->hidden_layer); $i++) $h[$i] = $this->forward($input,0,$i);
				//return $this->forward($h,1,0) >= $this->threshold ? 1 : 0;
				return $this->forward($h,1,0);
			}catch(Exception $e){
				echo "Error: " . $e->getMessage();
			}
		}

		public function get_weights(){
			return ["hidden_layer" => $this->hidden_layer, "output_layer" => $this->output_layer];
		}

		/*public function set_threshold($threshold = 0.5){
			if(!is_numeric($threshold)) $threshold = 0.5;
			if($threshold <= 0) $threshold = 0.01;
			if($threshold >= 1) $threshold = 0.99;
			$this->threshold = $threshold;
		}*/
	}
?>
