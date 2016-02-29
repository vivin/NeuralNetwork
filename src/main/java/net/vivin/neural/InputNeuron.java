package net.vivin.neural;

import net.vivin.neural.activators.PassThroughActivationStrategy;


/**
 * Created on 2/28/16 at 10:13 AM
 *
 * @author vivin
 */
public class InputNeuron extends Neuron implements SourceNeuron {

    private double output;

    public InputNeuron() {
        super(new PassThroughActivationStrategy());
    }

    void setOutput(double output) {
        this.output = output;
    }

    @Override
    public InputNeuron activate() {
        // do nothing
        return this;
    }

    @Override
    public double getOutput() {
        return output;
    }

    @Override
    public void addTargetWeightedInput(TargetNeuron neuron, WeightedInput weightedInput) {
        // do nothing
    }

    @Override
    public String toString() {
        return String.format("InputNeuron: [id = %s, input = %f]", id, output);
    }
}
