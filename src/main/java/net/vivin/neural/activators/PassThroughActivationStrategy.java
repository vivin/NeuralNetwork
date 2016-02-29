package net.vivin.neural.activators;

/**
 * Created on 2/23/16 at 8:35 PM
 *
 * @author vivin
 */
public class PassThroughActivationStrategy implements ActivationStrategy<PassThroughActivationStrategy> {
    @Override
    public double activate(double weightedSum) {
        return weightedSum;
    }

    @Override
    public double derivative(double weightedSum) {
        return 0;
    }

    @Override
    public PassThroughActivationStrategy copy() {
        return new PassThroughActivationStrategy();
    }
}
