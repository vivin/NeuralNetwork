package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created on 2/28/16 at 10:41 AM
 *
 * @author vivin
 */
public class HiddenNeuron extends Neuron implements TargetNeuron, SourceNeuron {

    private List<WeightedInput> weightedInputs;
    private Map<TargetNeuron, WeightedInput> targets;

    private Double weightedSum = null;
    private Double output = null;
    private Double derivative = null;

    private Double error = null;

    public HiddenNeuron(ActivationStrategy activationStrategy) {
        super(activationStrategy);

        this.weightedInputs = new ArrayList<>();
        this.targets = new HashMap<>();
    }

    @Override
    public void addTargetWeightedInput(TargetNeuron target, WeightedInput weightedInput) {
        targets.put(target, weightedInput);
    }

    @Override
    public void addWeightedInput(WeightedInput weightedInput) {
        weightedInputs.add(weightedInput);
    }

    @Override
    public void backpropagate(double expected) {
        error = derivative * targets.entrySet().stream()
            .mapToDouble(e -> e.getKey().getError() * e.getValue().getWeight())
            .sum();
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
    public HiddenNeuron activate() {
        calculateWeightedSum();
        output = activationStrategy.activate(weightedSum);
        derivative = activationStrategy.derivative(weightedSum);
        return this;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("HiddenNeuron: [\n")
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
