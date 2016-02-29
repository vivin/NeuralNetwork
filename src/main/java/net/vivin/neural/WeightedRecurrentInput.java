package net.vivin.neural;

/**
 * Created on 2/29/16 at 2:23 PM
 *
 * @author vivin
 */
public class WeightedRecurrentInput extends WeightedInput<LSTMNeuron> {
    public WeightedRecurrentInput(LSTMNeuron input, double weight) {
        super(input, weight);
    }

    @Override
    public double getInputValue() {
        return getInput().getPreviousOutput();
    }

    @Override
    public double getWeightedInputValue() {
        return getWeight() * this.getInputValue();
    }
}
