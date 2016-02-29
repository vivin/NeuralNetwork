package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;

import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Created on 2/28/16 at 10:31 AM
 *
 * @author vivin
 */
public class HiddenLayer extends Layer<HiddenNeuron> implements TargetLayer {
    private BiasNeuron bias = null;

    public HiddenLayer(int dimensions, ActivationStrategy activationStrategy) {
        super(Stream.generate(() -> new HiddenNeuron(activationStrategy)).limit(dimensions).collect(Collectors.toList()), dimensions);
    }

    public HiddenLayer(int dimensions, double bias, ActivationStrategy activationStrategy) {
        super(Stream.generate(() -> new HiddenNeuron(activationStrategy)).limit(dimensions).collect(Collectors.toList()), dimensions);
        this.bias = new BiasNeuron(bias);
    }

    @Override
    boolean hasBias() {
        return bias != null;
    }

    @Override
    BiasNeuron getBias() {
        return bias;
    }

    @Override
    public void backpropagate(double[] expected) {
        IntStream.range(0, neurons.size()).forEach(i -> neurons.get(i).backpropagate(0));
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
                WeightedInput weightedInput = new WeightedInput<>(source, (Math.random() * 1) - 0.5);
                target.addWeightedInput(weightedInput);
                source.addTargetWeightedInput(target, weightedInput);
            })
        );

        if(previous.hasBias()) {
            neurons.forEach(target -> target.addWeightedInput(new WeightedInput<>(previous.getBias(), (Math.random() * 1) - 0.5)));
        }
    }
}
