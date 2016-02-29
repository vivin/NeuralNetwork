package net.vivin.neural;


import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 12:42 PM
 */
public abstract class Layer<T extends Neuron> implements Serializable {

    protected int dimensions = 0;
    protected List<T> neurons;

    protected Layer previous = null;
    protected Layer next = null;

    public Layer(List<T> neurons, int dimensions) {
        this.dimensions = dimensions;
        this.neurons = neurons;
        this.previous = null;
    }

    public int getDimensions() {
        return dimensions;
    }

    Double[] getOutput() {
        return neurons.stream().map(Neuron::getOutput).collect(Collectors.toList()).toArray(new Double[neurons.size()]);
    }

    public void forward() {
        neurons.forEach(Neuron::activate);
    }

    abstract boolean hasBias();
    abstract BiasNeuron getBias();
}
