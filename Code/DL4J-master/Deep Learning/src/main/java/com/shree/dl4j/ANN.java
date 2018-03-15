package com.shree.dl4j;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ANN {

	private static Logger log = LoggerFactory.getLogger(ANN.class);

	public void ann_nw() throws IOException {

		// Initialize Variables

		int rngSeed = 123;
		int batchSize = 128;
		int numEpochs = 10;
		int outputNum = 10;

		final int numRows = 28;
		final int numColumns = 28;

		// Get the DataSetIterators:
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

		log.info("Build model....");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1).learningRate(0.006)
				.updater(Updater.NESTEROVS).regularization(true).l2(1e-4).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numRows * numColumns).nOut(1000).activation(Activation.RELU)
								.weightInit(WeightInit.XAVIER).build())
				.layer(1,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nIn(1000).nOut(outputNum)
								.activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		// Print score after every 50 iteration
		model.setListeners(new ScoreIterationListener(50));

		log.info("Train model....");
		for (int i = 0; i < numEpochs; i++) {
			model.fit(mnistTrain);
		}

		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum);

		while (mnistTest.hasNext()) {

			DataSet next = mnistTest.next();
			INDArray output = model.output(next.getFeatureMatrix());
			eval.eval(next.getLabels(), output);

		}

		log.info(eval.stats());
		log.info("****************Example finished********************");

	}

	public static void main(String[] args) throws IOException {

		ANN ann = new ANN();
		ann.ann_nw();

	}

}
