from torchinfo import summary

from TCMT import Circuit

def initialize_circuit(num_input_modes, num_output_modes, bias_cavities):
    """
    Initialize the circuit model.
    """
    num_modes = max(num_input_modes + bias_cavities, num_output_modes)
    circuit = Circuit.new(num_modes, num_input_modes)
    print("\nModel summary")
    summary(circuit)
    return circuit