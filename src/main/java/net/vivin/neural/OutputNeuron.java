package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.HashMap;
import java.util.Map;

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
        Neuron.updateSourceParameters(sources, getError(), learningRate, momentum);
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
}
