package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 1:39 PM
 */
public class NeuralNetwork implements Serializable {

    private final InputLayer inputLayer;
    private final List<HiddenLayer> hiddenLayers;
    private final OutputLayer outputLayer;

    private NeuralNetwork(Builder builder) {
        inputLayer = builder.inputLayer;
        hiddenLayers = builder.hiddenLayers;
        outputLayer = builder.outputLayer;

        if(hiddenLayers.isEmpty()) {
            outputLayer.connect(inputLayer);
        } else {
            hiddenLayers.get(0).connect(inputLayer);
            IntStream.range(1, hiddenLayers.size()).forEach(i -> hiddenLayers.get(i).connect(hiddenLayers.get(i - 1)));
            outputLayer.connect(hiddenLayers.get(hiddenLayers.size() - 1));
        }
    }

    public Double[] getOutput() {
        return outputLayer.getOutput();
    }

    public Double[] getOutput(double[] input) {
        if(input.length != inputLayer.getDimensions()) {
            throw new IllegalArgumentException("Input dimensions must match the dimensions of the input layer");
        }

        inputLayer.presentInput(input).forward();
        hiddenLayers.forEach(HiddenLayer::forward);
        outputLayer.forward();

        return outputLayer.getOutput();
    }

    public double train(double[] input, double[] expected, double learningRate, double momentum) {
        getOutput(input);

        outputLayer.backpropagate(expected);
        IntStream.range(0, hiddenLayers.size())
            .map(i -> 0 - i + hiddenLayers.size() - 1)
            .forEach(i -> hiddenLayers.get(i).backpropagate(expected));

        outputLayer.updateParameters(learningRate, momentum);
        IntStream.range(0, hiddenLayers.size())
            .map(i -> 0 - i + hiddenLayers.size() - 1)
            .forEach(i -> hiddenLayers.get(i).updateParameters(learningRate, momentum));

        Double[] output = getOutput(input);

        double error = 0;
        for (int i = 0; i < output.length; i++) {
            error += Math.pow(expected[i] - output[i], 2);
        }

        //System.out.println(toString());

        return error / 2;
    }

    public static class Builder {
        private final InputLayer inputLayer;
        private final List<HiddenLayer> hiddenLayers = new ArrayList<>();
        private final OutputLayer outputLayer;

        public Builder(int inputDimensions, int outputDimensions, ActivationStrategy outputActivationStrategy) {
            inputLayer = new InputLayer(inputDimensions);
            outputLayer = new OutputLayer(outputDimensions, outputActivationStrategy);
        }

        public Builder(int inputDimensions, double inputBias, int outputDimensions, ActivationStrategy outputActivationStrategy) {
            inputLayer = new InputLayer(inputDimensions, inputBias);
            outputLayer = new OutputLayer(outputDimensions, outputActivationStrategy);
        }

        public Builder addHiddenLayer(int dimensions, ActivationStrategy activationStrategy) {
            hiddenLayers.add(new HiddenLayer(dimensions, activationStrategy));
            return this;
        }

        public Builder addHiddenLayer(int dimensions, double bias, ActivationStrategy activationStrategy) {
            hiddenLayers.add(new HiddenLayer(dimensions, bias, activationStrategy));
            return this;
        }

        public NeuralNetwork build() {
            return new NeuralNetwork(this);
        }
    }

    /*
             .append(
                String.join("\n", neurons.stream().map(n ->
                    String.join("\n", Arrays.stream(n.toString().split("\n"))
                        .map(s -> String.format("\t%s", s))
                        .collect(Collectors.toList()))
                ).collect(Collectors.toList()))
            )
     */

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("NeuralNetwork: [\n")
            .append(
                String.join("\n", Arrays.stream(inputLayer.toString().split("\n"))
                    .map(s -> String.format("\t%s", s))
                    .collect(Collectors.toList())
                )
            )
            .append("\n")
            .append(
                String.join("\n", hiddenLayers.stream().map(h ->
                    String.join("\n", Arrays.stream(h.toString().split("\n"))
                        .map(s -> String.format("\t%s", s))
                        .collect(Collectors.toList())
                    )
                ).collect(Collectors.toList()))
            )
            .append("\n")
            .append(
                String.join("\n", Arrays.stream(outputLayer.toString().split("\n"))
                    .map(s -> String.format("\t%s", s))
                    .collect(Collectors.toList())
                )
            ).append("\n]");

        return builder.toString();
    }

/*
    public void reset() {
        for(Layer layer : layers) {
            for(Neuron neuron : layer.getNeurons()) {
                for(WeightedInput synapse : neuron.getInputs()) {
                    synapse.setWeight((Math.random() * 1) - 0.5);
                }
            }
        }
    }

    public void copyWeightsFrom(NeuralNetwork sourceNeuralNetwork) {
        if(layers.size() != sourceNeuralNetwork.layers.size()) {
            throw new IllegalArgumentException("Cannot copy weights. Number of layers do not match (" + sourceNeuralNetwork.layers.size() + " in source versus " + layers.size() + " in destination)");
        }

        int i = 0;
        for(Layer sourceLayer : sourceNeuralNetwork.layers) {
            Layer destinationLayer = layers.get(i);

            if(destinationLayer.getNeurons().size() != sourceLayer.getNeurons().size()) {
                throw new IllegalArgumentException("Number of neurons do not match in layer " + (i + 1) + "(" + sourceLayer.getNeurons().size() + " in source versus " + destinationLayer.getNeurons().size() + " in destination)");
            }

            int j = 0;
            for(Neuron sourceNeuron : sourceLayer.getNeurons()) {
                Neuron destinationNeuron = destinationLayer.getNeurons().get(j);

                if(destinationNeuron.getInputs().size() != sourceNeuron.getInputs().size()) {
                    throw new IllegalArgumentException("Number of inputs to neuron " + (j + 1) + " in layer " + (i + 1) + " do not match (" + sourceNeuron.getInputs().size() + " in source versus " + destinationNeuron.getInputs().size() + " in destination)");
                }

                int k = 0;
                for(WeightedInput sourceSynapse : sourceNeuron.getInputs()) {
                    WeightedInput destinationSynapse = destinationNeuron.getInputs().get(k);

                    destinationSynapse.setWeight(sourceSynapse.getWeight());
                    k++;
                }

                j++;
            }

            i++;
        }
    }*/
}
