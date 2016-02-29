package net.vivin;

import net.vivin.neural.NeuralNetwork;
import net.vivin.neural.activators.SigmoidActivationStrategy;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created on 2/28/16 at 1:44 PM
 *
 * @author vivin
 */
public class NNTest {
    /*
    NeuralNetwork xorNeuralNetwork = new NeuralNetwork("Trained XOR Network");

        Neuron inputBias = new Neuron(new SigmoidActivationStrategy());
        inputBias.setOutput(1);
        Layer inputLayer = new Layer(null, inputBias);

        Neuron a = new Neuron(new SigmoidActivationStrategy());
        a.setOutput(0);

        Neuron b = new Neuron(new SigmoidActivationStrategy());
        b.setOutput(0);

        inputLayer.addNeuron(a);
        inputLayer.addNeuron(b);

        Neuron bias = new Neuron(new SigmoidActivationStrategy());
        bias.setOutput(1);
        Layer hiddenLayer = new Layer(inputLayer, bias);

        Neuron hiddenA = new Neuron(new SigmoidActivationStrategy());
        Neuron hiddenB = new Neuron(new SigmoidActivationStrategy());

        hiddenLayer.addNeuron(hiddenA);
        hiddenLayer.addNeuron(hiddenB);

        Layer outputLayer = new Layer(hiddenLayer);
        Neuron xorNeuron = new Neuron(new SigmoidActivationStrategy());
        outputLayer.addNeuron(xorNeuron);

        xorNeuralNetwork.addLayer(inputLayer);
        xorNeuralNetwork.addLayer(hiddenLayer);
        xorNeuralNetwork.addLayer(outputLayer);

        return xorNeuralNetwork;
     */

    public static void main(String[] args) {
        /*
        NeuralNetwork or = new NeuralNetwork.Builder(2, 1, new ThresholdActivationStrategy(0.9))
            .withOutputWeights(new double[][]{{2, 2}})
            .build();
        or.presentInput(new double[]{0, 0});
        System.out.println(String.format("0 OR 0: %f", or.getOutput()[0]));

        or.presentInput(new double[]{0, 1});
        System.out.println(String.format("0 OR 1: %f", or.getOutput()[0]));

        or.presentInput(new double[]{1, 0});
        System.out.println(String.format("1 OR 0: %f", or.getOutput()[0]));

        or.presentInput(new double[]{1, 1});
        System.out.println(String.format("1 OR 1: %f", or.getOutput()[0])); */

        NeuralNetwork net = new NeuralNetwork.Builder(2, 1, 1, new SigmoidActivationStrategy())
            .addHiddenLayer(2, 1, new SigmoidActivationStrategy())
            .build();

        List<TrainingData> trainingDataList = new ArrayList<>();
        trainingDataList.add(new TrainingData(new double[]{0, 0}, new double[]{0}));
        trainingDataList.add(new TrainingData(new double[]{0, 1}, new double[]{1}));
        trainingDataList.add(new TrainingData(new double[]{1, 0}, new double[]{1}));
        trainingDataList.add(new TrainingData(new double[]{1, 1}, new double[]{0}));

        SecureRandom r = new SecureRandom();

        double iterations = 30000;
        double error;
        int i = 0;

        do {
            error = 0;
            Collections.shuffle(trainingDataList);
            for (TrainingData trainingData : trainingDataList) {
                error += net.train(trainingData.input, trainingData.output, 0.001, 0.9);
            }

            i++;
            System.out.println(String.format("Iteration #%d: %f", i, error));
        } while(i < iterations && error > 0.000001);

        net.getOutput(new double[]{0, 0});
        System.out.println(String.format("0 XOR 0: %f", net.getOutput()[0]));

        net.getOutput(new double[]{0, 1});
        System.out.println(String.format("0 XOR 1: %f", net.getOutput()[0]));

        net.getOutput(new double[]{1, 0});
        System.out.println(String.format("1 XOR 0: %f", net.getOutput()[0]));

        net.getOutput(new double[]{1, 1});
        System.out.println(String.format("1 XOR 1: %f", net.getOutput()[0]));
    }

    private static class TrainingData {
        private double[] input;
        private double[] output;

        public TrainingData(double[] input, double[] output) {
            this.input = input;
            this.output = output;
        }
    }
}
