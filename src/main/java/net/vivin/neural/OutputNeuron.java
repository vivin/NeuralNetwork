package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created on 2/28/16 at 12:01 PM
 *
 * @author vivin
 */
public class OutputNeuron extends Neuron implements TargetNeuron {
    private List<WeightedInput> weightedInputs;

    private Double weightedSum = null;
    private Double output = null;
    private Double derivative = null;

    private Double error = null;

    public OutputNeuron(ActivationStrategy activationStrategy) {
        super(activationStrategy);
        weightedInputs = new ArrayList<>();
    }

    @Override
    public void addWeightedInput(WeightedInput weightedInput) {
        weightedInputs.add(weightedInput);
    }

    @Override
    public void backpropagate(double expected) {
        this.error = derivative * (getOutput() - expected);
    }

    @Override
    public void updateParameters(double learningRate, double momentum) {
        Neuron.updateInputParameters(weightedInputs, error, learningRate, momentum);
    }

    @Override
    public double getError() {
        return error;
    }

    private void calculateWeightedSum() {
        weightedSum = weightedInputs.stream().mapToDouble(WeightedInput::getWeightedInputValue).sum();
    }

    public double getOutput() {
        return output;
    }

    @Override
    public OutputNeuron activate() {
        calculateWeightedSum();
        output = activationStrategy.activate(weightedSum);
        derivative = activationStrategy.derivative(weightedSum);
        return this;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("OutputNeuron: [\n")
            .append("\tid = ").append(id).append("\n")
            .append("\tweightedInputs = [\n")
            .append(
                String.join("\n", weightedInputs.stream().map(s -> String.format("\t\t[id = %s, weight = %f]", s.getInput().getId(), s.getWeight())).collect(Collectors.toList()))
            )
            .append("\n\t]\n")
            .append("\tweightedSum = ").append(weightedSum).append("\n")
            .append("\toutput = ").append(output).append("\n")
            .append("\tderivative = ").append(derivative).append("\n")
            .append("\terror = ").append(error).append("\n")
            .append("]");

        return builder.toString();
    }
}
