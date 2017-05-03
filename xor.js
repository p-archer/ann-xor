const learningRatio = 0.001;
const minCorrect = 4;
const maxEpochs = 100000;
const operator = 'xor';
const activationFunction = 'sigmoid';

function start() {
	//create network topology
	let topology = [
		[{bias: 1}, {bias: 1}, {bias: 1}, {bias: 1}],
		// [{bias: 0}, {bias: 1}],
		[{bias: 0}]
	];
	// let topology = [[{bias: 1}]];
	let inputCount = 2;
	let network = createNetwork(topology, inputCount);
	show(network);

	train(network);
}

function train(network) {
	let logArray = [];
	let epochCount = 0;
	let consecutiveCorrect = 0;
	while (epochCount < maxEpochs && consecutiveCorrect < minCorrect) {
		//create input and expected output
		let input;
		switch (epochCount % 4) {
		case 0: input = [0, 0]; break;
		case 1: input = [0, 1]; break;
		case 2: input = [1, 0]; break;
		case 3: input = [1, 1]; break;
		}

		switch (operator) {
		case 'or': output = input[0] | input[1]; break;
		case 'and': output = input[0] & input[1]; break;
		case 'nor': output = (input[0] | input[1]) ^ 1; break;
		case 'nand': output = (input[0] & input[1]) ^ 1; break;
		case 'xor': output = input[0] ^ input[1]; break;
		}

		//feed data until learning is done
		let resultArray = process(network, input);
		let result = resultArray[resultArray.length-1][0];
		result = result > 0.5 ? 1 : 0;

		//learn
		let error = output - result;
		learn(network, resultArray.slice(), error);

		if (error === 0) {
			consecutiveCorrect++;
		} else {
			consecutiveCorrect = 0;
		}

		//log
		logArray.push(resultArray);
		while (logArray.length > 4) {
			logArray.shift();
		}

		epochCount++;
	}

	console.log();
	console.log('finished in', epochCount, 'epochs');
	for (let i=0; i<logArray.length; i++) {
		let inputArray = [];
		for (let j=0; j<logArray[i].length; j++) {
			inputArray.push(logArray[i][j]);
		}
		console.log('epoch', epochCount-logArray.length+1+i, 'res', inputArray.join(', '));
	}
	show(network);
}

function createNetwork(topology, initialInputCount) {
	let net = [];

	for (let i=0; i<topology.length; i++) {
		let layer = [];

		for (let j=0; j<topology[i].length; j++) {
			let neuron = { weights: [], bias: topology[i][j].bias };

			let inputCount = i === 0 ? initialInputCount + 1 : topology[i-1].length + 1;
			for (let k=0; k<inputCount; k++) {
				neuron.weights.push(2 * Math.random() - 1);
			}

			layer.push(neuron);
		}

		net.push(layer);
	}

	return net;
}

function process(net, input) {
	let outputs = [input];

	for (let i=0; i<net.length; i++) {
		let layer = net[i];
		outputs.push([]);
		for (let j=0; j<net[i].length; j++) {
			let neuron = layer[j];
			let inputVector = outputs[i].slice();
			inputVector.push(neuron.bias);

			let sig = 0;
			for (let k=0; k<inputVector.length; k++) {
				sig += inputVector[k] * neuron.weights[k];
			}

			switch (activationFunction.toLowerCase()) {
			case 'linear':
			case 'perceptron': sig = sig > 0.5 ? 1 : 0; break;
			case 'sigmoid':
			case 'logistic': sig = 1 / (1 + Math.exp(-sig)); break;
			case 'rectified':
			case 'relu': sig = sig > 0 ? sig : 0; break;
			default: throw('unknown activation function');
			}

			outputs[i+1].push(sig);
		}
	}

	return outputs;
}

function learn(net, inputs, error) {
	if (error !== 0) {
		for (let i=0; i<net.length; i++) { //foreach layer
			let layer = net[i];
			for (let j=0; j<layer.length; j++) { // foreach neuron
				let neuron = layer[j];
				let inputVector = inputs[i].slice();
				inputVector.push(neuron.bias);
				for (let k=0; k<neuron.weights.length; k++) {
					neuron.weights[k] += inputVector[k] * error * learningRatio;
				}
			}
		}
	}
}

function show(network) {
	for (let i=0; i<network.length; i++) {
		console.log('layer', i);
		for (let j=0; j<network[i].length; j++) {
			console.log('\tneuron', j, 'bias', network[i][j].bias);
			let weights = [];
			for (let k=0; k<network[i][j].weights.length; k++) {
				weights.push(network[i][j].weights[k].toFixed(4));
			}
			console.log('\t\t', weights.join(', '));
		}
	}
}

start();
