package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created on 2/28/16 at 12:01 PM
 *
 * @author vivin
 */
public class OutputNeuron extends Neuron implements TargetNeuron {
    private Map<Neuron, Synapse> sources;

    private Double weightedSum = null;
    private Double output = null;
    private Double derivative = null;

    private Double error = null;

    public OutputNeuron(ActivationStrategy activationStrategy) {
        super(activationStrategy);
        sources = new HashMap<>();
    }

    @Override
    public void addSourceSynapse(Synapse synapse) {
        sources.put(synapse.getSource(), synapse);
    }

    public Synapse getSynapseForSource(Neuron neuron) {
        return sources.get(neuron);
    }

    @Override
    public void backpropagate(double expected) {
        this.error = derivative * (getOutput() - expected);
    }

    @Override
    public void updateParameters(double learningRate, double momentum) {
        Neuron.updateSourceParameters(sources, error, learningRate, momentum);
    }

    @Override
    public double getError() {
        return error;
    }

    private void calculateWeightedSum() {
        weightedSum = sources.values().stream().mapToDouble(s -> s.getWeight() * s.getSource().getOutput()).sum();
    }

    public double getOutput() {
        return output;
    }

    @Override
    public void activate() {
        calculateWeightedSum();
        output = activationStrategy.activate(weightedSum);
        derivative = activationStrategy.derivative(weightedSum);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("OutputNeuron: [\n")
            .append("\tid = ").append(id).append("\n")
            .append("\tsources = [\n")
            .append(
                String.join("\n", sources.values().stream().map(s -> String.format("\t\t[id = %s, weight = %f]", s.getSource().getId(), s.getWeight())).collect(Collectors.toList()))
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
