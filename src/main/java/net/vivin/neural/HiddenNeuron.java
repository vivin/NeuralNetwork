package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.HashMap;
import java.util.Map;

/**
 * Created on 2/28/16 at 10:41 AM
 *
 * @author vivin
 */
public class HiddenNeuron extends Neuron implements TargetNeuron, SourceNeuron {

    private Map<Neuron, Synapse> sources;
    private Map<TargetNeuron, Synapse> targets;

    private Double weightedSum = null;
    private Double output = null;
    private Double derivative = null;

    private Double error = null;

    public HiddenNeuron(ActivationStrategy activationStrategy) {
        super(activationStrategy);

        this.sources = new HashMap<>();
        this.targets = new HashMap<>();
    }

    @Override
    public void addTargetSynapse(TargetNeuron target, Synapse synapse) {
        targets.put(target, synapse);
    }

    @Override
    public void addSourceSynapse(Synapse synapse) {
        sources.put(synapse.getSource(), synapse);
    }

    @Override
    public void backpropagate(double expected) {
        error = derivative * targets.entrySet().stream()
            .mapToDouble(e -> e.getKey().getError() * e.getValue().getWeight())
            .sum();
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
