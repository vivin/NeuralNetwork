package net.vivin.neural;

/**
 * Created on 2/28/16 at 4:57 PM
 *
 * @author vivin
 */
public interface SourceNeuron {
    void addTargetSynapse(TargetNeuron neuron, Synapse synapse);
}
