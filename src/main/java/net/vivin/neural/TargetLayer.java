package net.vivin.neural;

/**
 * Created on 2/28/16 at 12:46 PM
 *
 * @author vivin
 */
interface TargetLayer {
    <V extends  Neuron & SourceNeuron, U extends Layer<V>> void connect(U previous);
    void backpropagate(double[] expected);
    void updateParameters(double learningRate, double momentum);
}
