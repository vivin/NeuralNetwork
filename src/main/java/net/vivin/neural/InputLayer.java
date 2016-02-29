package net.vivin.neural;

import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Created on 2/28/16 at 10:09 AM
 *
 * @author vivin
 */
public class InputLayer extends Layer<InputNeuron> {

    private BiasNeuron bias = null;

    public InputLayer(int dimensions) {
        super(Stream.generate(InputNeuron::new).limit(dimensions).collect(Collectors.toList()), dimensions);
    }

    public InputLayer(int dimensions, double bias) {
        this(dimensions);
        this.bias = new BiasNeuron(bias);
    }

    public InputLayer presentInput(double[] input) {
        IntStream.range(0, input.length).forEach(i -> neurons.get(i).setOutput(input[i]));
        return this;
    }

    @Override
    boolean hasBias() {
        return bias != null;
    }

    @Override
    BiasNeuron getBias() {
        return bias;
    }
}
