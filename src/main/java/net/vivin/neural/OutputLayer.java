package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Created on 2/28/16 at 1:05 PM
 *
 * @author vivin
 */
public class OutputLayer extends Layer<OutputNeuron> implements TargetLayer {
    public OutputLayer(int dimensions, ActivationStrategy activationStrategy) {
        super(Stream.generate(() -> new OutputNeuron(activationStrategy)).limit(dimensions).collect(Collectors.toList()), dimensions);
    }

    @Override
    boolean hasBias() {
        return false;
    }

    @Override
    BiasNeuron getBias() {
        return null;
    }

    @Override
    public void backpropagate(double[] expected) {
        IntStream.range(0, neurons.size()).forEach(i -> neurons.get(i).backpropagate(expected[i]));
    }

    @Override
    public void updateParameters(double learningRate, double momentum) {
        IntStream.range(0, neurons.size()).forEach(i -> neurons.get(i).updateParameters(learningRate, momentum));
    }

    @Override
    public <V extends Neuron & SourceNeuron, U extends Layer<V>> void connect(U previous) {
        this.previous = previous;
        previous.next = this;

        previous.neurons.forEach(source ->
            neurons.forEach(target -> {
                Synapse synapse = new Synapse<>(source, (Math.random() * 1) - 0.5);
                target.addSourceSynapse(synapse);
                source.addTargetSynapse(target, synapse);
            })
        );

        if(previous.hasBias()) {
            neurons.forEach(target -> target.addSourceSynapse(new Synapse<>(previous.getBias(), (Math.random() * 1) - 0.5)));
        }
    }
}
