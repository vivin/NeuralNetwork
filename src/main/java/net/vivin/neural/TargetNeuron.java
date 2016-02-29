package net.vivin.neural;

/**
 * Created on 2/28/16 at 11:05 AM
 *
 * @author vivin
 */
public interface TargetNeuron {
    void addSourceSynapse(Synapse synapse);
    double getError();
    void backpropagate(double expected);
    void updateParameters(double learningRate, double momentum);
}
