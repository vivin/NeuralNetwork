package net.vivin.neural;

/**
 * Created on 2/28/16 at 12:18 PM
 *
 * @author vivin
 */
public class BiasNeuron extends InputNeuron {
    private BiasNeuron() {
    }

    public BiasNeuron(double bias) {
        super();
        this.setOutput(bias);
    }
}
