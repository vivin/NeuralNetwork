package net.vivin.neural;

import net.vivin.neural.activators.ActivationStrategy;
import net.vivin.neural.activators.HyperbolicTangentActivationStrategy;
import net.vivin.neural.activators.SigmoidActivationStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

/**
 * Created on 2/29/16 at 12:02 PM
 *
 * @author vivin
 */
public class LSTMNeuron extends Neuron implements SourceNeuron, TargetNeuron {

    private List<WeightedInput> weightedInputs;
    private Map<TargetNeuron, WeightedInput> targets;

    private final HiddenNeuron inputNode;
    private final Gate inputGate;
    private final Gate forgetGate;
    private final Gate outputGate;

    private double output;
    private double state;

    private double inputNodeValue;
    private double inputGateValue;
    private double forgetGateValue;
    private double outputGateValue;

    private Stack<Double> previousStates = new Stack<>();
    private Stack<Double> previousOutputs = new Stack<>();

    private Stack<Double> errors = new Stack<>();

    public LSTMNeuron() {
        super(new HyperbolicTangentActivationStrategy());
        previousStates.push(0.0);
        previousOutputs.push(0.0);

        inputNode = new HiddenNeuron(new HyperbolicTangentActivationStrategy());
        inputGate = new Gate("input");
        forgetGate = new Gate("forget");
        outputGate = new Gate("output");

        weightedInputs.add(new WeightedRecurrentInput(this, (Math.random() * 1.0) - 0.5));
    }

    @Override
    public LSTMNeuron activate() {
        inputNodeValue = inputNode.activate().getOutput();
        inputGateValue = inputGate.activate();
        forgetGateValue = forgetGate.activate();
        outputGateValue = outputGate.activate();

        state = (inputNodeValue * inputGateValue) + (getPreviousState() * forgetGateValue);
        output = activationStrategy.activate(state) * outputGateValue;

        return this;
    }

    @Override
    public double getOutput() {
        return output;
    }

    public double getPreviousOutput() {
        return previousOutputs.peek();
    }

    private double getPreviousState() {
        return previousStates.peek();
    }

    @Override
    public void addTargetWeightedInput(TargetNeuron neuron, WeightedInput weightedInput) {
        targets.put(neuron, weightedInput);
        inputNode.addTargetWeightedInput(neuron, weightedInput);
    }

    @Override
    public void addWeightedInput(WeightedInput weightedInput) {
        weightedInputs.add(weightedInput);
        inputNode.addWeightedInput(weightedInput);

        inputGate.addWeightedInput(new WeightedInput<>(weightedInput.getInput(), weightedInput.getWeight()));
        forgetGate.addWeightedInput(new WeightedInput<>(weightedInput.getInput(), weightedInput.getWeight()));
        outputGate.addWeightedInput(new WeightedInput<>(weightedInput.getInput(), weightedInput.getWeight()));
    }

    @Override
    public double getError() {
        return 0;
    }

    @Override
    public void backpropagate(double expected) {

    }

    @Override
    public void updateParameters(double learningRate, double momentum) {

    }

    private class Gate {
        private String name;

        private List<WeightedInput> weightedInputs;
        private double previousOutputWeight = (Math.random() * 1.0) - 0.5;

        private ActivationStrategy activationStrategy;

        public Gate(String name) {
            this.name = name;
            this.activationStrategy = new SigmoidActivationStrategy();
            this.weightedInputs = new ArrayList<>();
        }

        public double activate() {
            return activationStrategy.activate(
                weightedInputs.stream().mapToDouble(WeightedInput::getWeightedInputValue).sum() + (previousOutputWeight * getPreviousOutput())
            );
        }

        public void addWeightedInput(WeightedInput weightedInput) {
            weightedInputs.add(weightedInput);
        }

        public double getPreviousOutputWeight() {
            return previousOutputWeight;
        }

        public void setPreviousOutputWeight(double previousOutputWeight) {
            this.previousOutputWeight = previousOutputWeight;
        }
    }
}
