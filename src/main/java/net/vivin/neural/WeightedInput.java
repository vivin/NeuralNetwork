package net.vivin.neural;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 12:03 PM
 */
public class WeightedInput<T extends Neuron> implements Serializable {

    private T input;

    private double weight;
    private double previousDelta = 0;

    public WeightedInput(T input, double weight) {
        this.input = input;
        this.weight = weight;
    }

    public T getInput() {
        return input;
    }

    public double getInputValue() {
        return input.getOutput();
    }

    public double getWeightedInputValue() {
        return weight * input.getOutput();
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getPreviousDelta() {
        return previousDelta;
    }

    public void setPreviousDelta(double previousDelta) {
        this.previousDelta = previousDelta;
    }
}
