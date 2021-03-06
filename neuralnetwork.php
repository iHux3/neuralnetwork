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

		public function __construct(int $input_neurons = 2,int $hidden_neurons = 2){
			if($input_neurons < 1) $input_neurons = 1;
			if($hidden_neurons < 1) $hidden_neurons = 1;
			$this->il = round($input_neurons);
			$this->hl = round($hidden_neurons);

			$output_weights = [];
			for($i = 0; $i < $this->hl; $i++){
				$hidden_weights = [];
				for($j = 0; $j < $this->il; $j++){
					$hidden_weights[$j] = $this->random_weight();
				}
				$this->hidden_layer[$i] = ["weights" => $hidden_weights, "bias" => $this->random_weight()];
				$output_weights[$i] = $this->random_weight();
			}
			$this->output_layer = [
				["weights" => $output_weights, "bias" => $this->random_weight()]
			];
		}

		private function random_weight() : float{
			return rand(0,99) / 100;
		}

		private function sigmoid(float $f) : float{
			return 1 / (1 + pow(M_E,-$f));
		}

		private function forward(array $input,int $type,int $id) : float{
			$layer = $type == 0 ? $this->hidden_layer : $this->output_layer;
			$result = 0;
			for($i = 0; $i < count($input); $i++){
				$result += $input[$i] * $layer[$id]["weights"][$i];
			}
			$result += $this->hidden_layer[$id]["bias"];

			return self::sigmoid($result);
		}

		private function backward(float $o, float $target, array $h, array $input) : void{
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

		private function error(float $target, float $o) : float{
			return $this->learning_rate * pow($target - $o,2);
		}

		private function calc_threshold(array $targets,array $inputs) : void{
			$t0 = [];
			$t1 = [];
			for($i = 0; $i < count($targets); $i++){
				$p = $this->predict($inputs[$i]);
				if($targets[$i] == 0) $t0[] = $p; else $t1[] = $p;
			}
			$max = max($t0);
			$min = min($t1);
			$this->threshold = $min > $max ? ($min - $max) / 2 : ($max - $min) / 2;
		}

		private function check_learn(array $targets,array $inputs, array $params) : void{
			$zero = false;
			$one = false;
			foreach($targets as $target){
				if(is_array($target)) throw new Exception("@p0 must have exactly one column");
				if(!is_numeric($target)) throw new Exception("@p0 non-numeric value detected");
				if($target != 0 && $target != 1) throw new Exception("@p0 values must be 0 or 1");
				if($target == 0) $zero = true; else $one = true;
			}
			if($zero != true || $one != true) throw new Exception("@p0 must include at least 1 zero and 1 one");
			if(count($targets) != count($inputs)) throw new Exception("@p0 must contain the same number of rows as @p1");

			foreach($inputs as $input){
				if(!is_array($input)) throw new Exception("@p1 must be a 2d array");
				if(count($input) != $this->il) throw new Exception("@p1 must contain $this->il columns");
				foreach($input as $i){
					if(!is_numeric($i)) throw new Exception("@p1 non-numeric value detected");
				}
			}
			if($params["err_min"] <= 0) throw new Exception("@p2['err_min'] must be greater than zero");
			if($params["max_loops"] <= 0) throw new Exception("@p2['max_loops'] must be greater than zero");
			if($params["learning_rate"] <= 0) throw new Exception("@p2['learning_rate'] must be greater than zero");
		}

		private function check_predict(array $input){
			if(count($input) != $this->il) throw new Exception("@p0 must contain $this->il numbers");
			foreach($input as $i){
				if(!is_numeric($i)) throw new Exception("@p0 non-numeric value detected");
			}
		}

		private function check_modify_layer(array $data,int $type) : void{
			if($type == 0){
				if(count($data) != $this->hl) throw new Exception("@p0 must contain $this->hl rows");
			}else{
				if(count($data) != 1) throw new Exception("@p0 must contain exactly 1 row");
			}

			foreach($data as $d){
				if(!is_array($d)) throw new Exception("@p0 must be a 2d array");
				if(!isset($d["weights"])) throw new Exception("@p0 key weights missing");
				if($type == 0){
					if(count($d["weights"]) != $this->il) throw new Exception("@p0 weights must contain ". $this->il  . " numbers");
				}else{
					if(count($d["weights"]) != $this->hl) throw new Exception("@p0 weights must contain ". $this->hl  . " numbers");
				}

				foreach($d["weights"] as $i){
					if(!is_numeric($i)) throw new Exception("@p0 non-numeric value detected");
				}
				if(!isset($d["bias"])) throw new Exception("@p0 key bias missing");
				if(!is_numeric($d["bias"])) throw new Exception("@p0 non-numeric value detected");
			}
		}

		public function learn(array $targets,array $inputs,array $params = []) : void{
			try{
				$err_min = $params["err_min"] ?? 0.001;
				$max_loops = $params["max_loops"] ?? 10000;
				$learning_rate = $params["learning_rate"] ?? 0.5;

				$this->check_learn($targets,$inputs,["err_min" => $err_min, "max_loops" => $max_loops, "learning_rate" => $learning_rate]);
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
				}while($errSum > $err_min && $this->total_loops < $max_loops);
				$this->calc_threshold($targets,$inputs);

			}catch(Exception $e){
				echo "learn<" . $e->getMessage() . ">" . PHP_EOL;
			}
		}

		public function predict(array $input,int $type = 0) : float{
			try{
				$this->check_predict($input);
				$h = [];
				for($i = 0; $i < count($this->hidden_layer); $i++) $h[$i] = $this->forward($input,0,$i);
				$result = $this->forward($h,1,0);
				if($type == 0){
					return $result;
				}else{
					return $result >= $this->threshold ? 1 : 0;
				}
			}catch(Exception $e){
				echo "predict<" . $e->getMessage() . ">" . PHP_EOL;
			}
		}

		public function modify_layer(array $data,array $targets,array $inputs,int $type = 0) : void{
			try{
				$this->check_modify_layer($data,$type);
				if($type == 0)	$this->hidden_layer = $data; else $this->output_layer = $data;
				$this->calc_threshold($targets,$inputs);
			}catch(Exception $e){
				echo "modify_layer<" . $e->getMessage() . ">". PHP_EOL;
			}
		}

		public function get_structure() : string{
			$return = "\n";
			for($i = 0; $i < $this->hl; $i++){
				$return .= implode(",",$this->hidden_layer[$i]["weights"]) . "," . $this->hidden_layer[$i]["bias"];
				$return .= "\n";
			}
			$return .= "\n";
			$return .= implode(",",$this->output_layer[0]["weights"]) . "," . $this->output_layer[0]["bias"];
			return $return;
		}

		public function get_threshold() : float{
			return $this->threshold;
		}
	}
?>
