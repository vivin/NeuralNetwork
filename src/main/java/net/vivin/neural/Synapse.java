package net.vivin.neural;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 12:03 PM
 */
public class Synapse<T extends Neuron> implements Serializable {

    private T source;

    private double weight;
    private double previousDelta = 0;

    public Synapse(T source, double weight) {
        this.source = source;
        this.weight = weight;
    }

    public T getSource() {
        return source;
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
