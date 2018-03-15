package com.shree.dl4j;

import java.util.Map;
import java.io.IOException;
import java.util.HashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;

import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

public class CNN {

	private static final Logger log = LoggerFactory.getLogger(CNN.class);

	public void lenet() throws IOException {

		// Initialize variables

		int seed = 123;

		int nEpochs = 5;
		int iterations = 1;
		int nChannels = 1;
		int batchSize = 64;

		int outputNum = 10;

		// Initialize Test and Training Data
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);
		log.info("Load data....");

		// Initialize learning rate scheduler
		Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
		lrSchedule.put(0, 0.01);
		lrSchedule.put(1000, 0.005);
		lrSchedule.put(3000, 0.001);

		// Construct CNN model - LENET
		log.info("Build model....");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
				.regularization(true).l2(0.0005).learningRate(.01).learningRateDecayPolicy(LearningRatePolicy.Schedule)
				.learningRateSchedule(lrSchedule).weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).list()
				.layer(0,
						new ConvolutionLayer.Builder(5, 5).nIn(nChannels).stride(1, 1).nOut(20)
								.activation(Activation.IDENTITY).build())
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(2,
						new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(50).activation(Activation.IDENTITY)
								.build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
				.layer(5,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
								.activation(Activation.SOFTMAX).build())
				.setInputType(InputType.convolutionalFlat(28, 28, 1)) 
				.backprop(true).pretrain(false).build();

		// Train and evaluate CNN model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(50));

		for (int i = 0; i < nEpochs; i++) {

			model.fit(mnistTrain);
			log.info("*** Completed epoch {} ***", i);

			log.info("Evaluate model....");
			Evaluation eval = model.evaluate(mnistTest);

			log.info(eval.stats());
			mnistTest.reset();

		}

		log.info("****************Example finished********************");

	}

	public static void main(String[] args) throws IOException {
		CNN cnn = new CNN();
		cnn.lenet();
	}

}
