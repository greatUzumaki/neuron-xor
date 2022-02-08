class Input {
  constructor(neuron, weight) {
    this.neuron = neuron;
    this.weight = weight;
  }
}

class Neuron {
  constructor(layer, previousLayer) {
    this._layer = layer;
    this.inputs = previousLayer
      ? previousLayer.neurons.map(
          (neuron) => new Input(neuron, Math.random() - 0.5)
        )
      : [0];
  }

  // Вычисляемый флаг, который будет говорить нам о том, что это нейрон входного слоя. Некоторые операции производятся во всех слоях, кроме входного, так что он пригодится нам в дальнейшем.
  get $isFirstLayerNeuron() {
    return !(this.inputs[0] instanceof Input);
  }

  // Предоставляет сумму всех входных сигналов (значений, на выходе нейронов предыдущего слоя) с примененными к ним весами.
  get inputSum() {
    return this.inputs.reduce((sum, input) => {
      return sum + input.neuron.value * input.weight;
    }, 0);
  }

  // Выходное значение нейрона. Для всех слоев, кроме входного, это значение активационной функции от inputSum.
  get value() {
    return this.$isFirstLayerNeuron
      ? this.inputs[0]
      : this._layer._network.activationFunction(this.inputSum);
  }

  // Нужен для того, чтобы устанавливать входные значения для нейронов первого слоя.
  set input(val) {
    if (!this.$isFirstLayerNeuron) {
      return;
    }

    this.inputs[0] = val;
  }

  // Рассчитываем дельту весов, и для каждого входа нейрона пересчитываем их значения, а также устанавливаем error для нейрона предыдущего слоя. Таким образом, установив ошибку на выходе сети, мы инициируем пересчет весов всех нейронных связей.
  set error(error) {
    if (this.$isFirstLayerNeuron) {
      return;
    }

    const wDelta =
      error * this._layer._network.derivativeFunction(this.inputSum);

    this.inputs.forEach((input) => {
      input.weight -=
        input.neuron.value * wDelta * this._layer._network.learningRate;
      input.neuron.error = input.weight * wDelta;
    });
  }
}
class Layer {
  constructor(neuronsCount, previousLayer, network) {
    this._network = network;
    this.neurons = [];
    for (let i = 0; i < neuronsCount; i++) {
      this.neurons.push(new Neuron(this, previousLayer));
    }
  }

  // Вычисляемый флаг, который говорит о том, входной ли это слой
  get $isFirstLayer() {
    return this.neurons[0].$isFirstLayerNeuron;
  }

  // Сеттер, который после проверок на то, что мы вызываем его на входном слое и на вход получаем массив нужной длины, устанавливает входные данные. Он нужен для того, чтобы заполнить значениями входной слой.
  set input(val) {
    if (!this.$isFirstLayer) {
      return;
    }

    if (!Array.isArray(val)) {
      return;
    }

    if (val.length !== this.neurons.length) {
      return;
    }

    val.forEach((v, i) => (this.neurons[i].input = v));
  }
}

class Network {
  static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  static sigmoidDerivative(x) {
    return Network.sigmoid(x) * (1 - Network.sigmoid(x));
  }

  constructor(
    inputSize,
    outputSize,
    hiddenLayersCount = 1,
    learningRate = 0.5
  ) {
    this.activationFunction = Network.sigmoid;
    this.derivativeFunction = Network.sigmoidDerivative;
    this.learningRate = learningRate;

    this.layers = [new Layer(inputSize, null, this)];

    for (let i = 0; i < hiddenLayersCount; i++) {
      const layerSize = Math.min(
        inputSize * 2 - 1,
        Math.ceil((inputSize * 2) / 3 + outputSize)
      );
      this.layers.push(
        new Layer(layerSize, this.layers[this.layers.length - 1], this)
      );
    }

    this.layers.push(
      new Layer(outputSize, this.layers[this.layers.length - 1], this)
    );
  }

  //Просто сахар, по сути вызывает одноименный сеттер входного слоя
  set input(val) {
    this.layers[0].input = val;
  }

  //Выдает нам предсказание нейросети в виде массива значений нейронов выходного слоя на основе установленных входных данных.
  get prediction() {
    return this.layers[this.layers.length - 1].neurons.map(
      (neuron) => neuron.value
    );
  }

  trainOnce(dataSet) {
    if (!Array.isArray(dataSet)) {
      return;
    }

    dataSet.forEach((dataCase) => {
      const [input, expected] = dataCase;

      this.input = input;
      this.prediction.forEach((r, i) => {
        this.layers[this.layers.length - 1].neurons[i].error = r - expected[i];
      });
    });
  }

  train(dataSet, epochs = 100000) {
    return new Promise((resolve) => {
      for (let i = 0; i < epochs; i++) {
        this.trainOnce(dataSet);
      }
      resolve();
    });
  }
}

const network = new Network(2, 1);

// Набор данных для обучения
const data = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]],
];

network.train(data).then(() => {
  const testData = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];

  testData.forEach((input, index) => {
    network.input = input;
    console.log(`${input[0]} XOR ${input[1]} = ${network.prediction}`);
  });
});
