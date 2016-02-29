package net.vivin.neural.activators;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/5/11
 * Time: 3:04 PM
 */
public interface ActivationStrategy<T extends ActivationStrategy<T>> {
    double activate(double weightedSum);
    double derivative(double weightedSum);
    T copy();
}
