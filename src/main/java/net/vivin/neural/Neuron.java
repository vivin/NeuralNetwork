package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.io.Serializable;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 11:52 AM
 */
public abstract class Neuron implements Serializable {

    protected UUID id;
    protected ActivationStrategy activationStrategy;

    public Neuron(ActivationStrategy activationStrategy) {
        this.id = UUID.randomUUID();
        this.activationStrategy = activationStrategy;
    }

    public UUID getId() {
        return id;
    }

    public abstract Neuron activate();
    public abstract double getOutput();

    protected static void updateInputParameters(List<WeightedInput> weightedInputs, double error, double learningRate, double momentum) {
        weightedInputs.forEach(weightedInput -> {
            double delta = learningRate * error * weightedInput.getInputValue();
            delta += momentum * weightedInput.getPreviousDelta();

            weightedInput.setPreviousDelta(delta);
            weightedInput.setWeight(weightedInput.getWeight() - delta);
        });
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Neuron)) return false;

        Neuron neuron = (Neuron) o;

        return id.equals(neuron.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }
}
