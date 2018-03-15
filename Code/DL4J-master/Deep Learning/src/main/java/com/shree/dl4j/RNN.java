package com.shree.dl4j;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;

public class RNN {

	private static final int EPOCHS = 10;
	private static final char[] LEARNSTRING = "Smile sets everything straight".toCharArray();

	private static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<Character>();

	// RNN dimensions
	private static final int HIDDEN_LAYER_COUNT = 2;
	private static final int HIDDEN_LAYER_WIDTH = 50;

	public void characterRecognition() {

		// create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
		LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<Character>();

		for (char c : LEARNSTRING)
			LEARNSTRING_CHARS.add(c);

		LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);

		// Initialize common parameters
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.seed(123);
		builder.biasInit(0);
		builder.iterations(10);
		builder.miniBatch(false);
		builder.learningRate(0.001);
		builder.updater(Updater.RMSPROP);
		builder.weightInit(WeightInit.XAVIER);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

		ListBuilder listBuilder = builder.list();

		// Build Basic RNN Architecture

		// 1. Hidden Layer
		for (int i = 0; i < HIDDEN_LAYER_COUNT; i++) {
			GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? LEARNSTRING_CHARS.size() : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.activation(Activation.TANH);
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		// 2. Output Layer
		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.nOut(LEARNSTRING_CHARS.size());
		listBuilder.layer(HIDDEN_LAYER_COUNT, outputLayerBuilder.build());

		listBuilder.pretrain(false);
		listBuilder.backprop(true);

		// Create Network Using DL4J
		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		// Create Training Data
		INDArray input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		INDArray labels = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);

		// Loop through sample-sentence - "Smile sets everything straight"-
		int samplePos = 0;
		for (char currentChar : LEARNSTRING) {
			char nextChar = LEARNSTRING[(samplePos + 1) % (LEARNSTRING.length)];
			input.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos }, 1);
			labels.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos }, 1);
			samplePos++;
		}

		// Initialize training data
		DataSet trainingData = new DataSet(input, labels);

		// Train RNN model
		for (int epoch = 0; epoch < EPOCHS; epoch++) {

			System.out.println("Epoch " + epoch);

			// train data
			net.fit(trainingData);

			// clear current stance from the last example
			net.rnnClearPreviousState();

			// put the first character into the RNN as an initialisation input
			INDArray testInit = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
			testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0]), 1);

			// the output shows what the net thinks what should come next
			INDArray output = net.rnnTimeStep(testInit);

			// Net should guess LEARNSTRING.length more characters
			for (char x : LEARNSTRING) {

				// Neuron with the highest output has the highest chance to get chosen
				int sampledCharacterIdx = Nd4j.getExecutioner().exec(new IMax(output), 1).getInt(0);

				// Print chosen output
				System.out.print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx));

				// Use last output as input
				INDArray nextInput = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
				nextInput.putScalar(sampledCharacterIdx, 1);
				output = net.rnnTimeStep(nextInput);

			}
			System.out.print("\n");
		}

	}

	public static void main(String[] args) {
		RNN rnn = new RNN();
		rnn.characterRecognition();
	}

}
