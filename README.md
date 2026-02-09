# MLP (Multi-Layer Perceptron) in SystemVerilog

## Project Overview

This project implements a **Generalized Multi-Layer Perceptron (MLP) neural network in SystemVerilog**. The goal is to build a synthesizable, scalable hardware accelerator for neural network inference with:

- ✅ **Flexible Architecture**: Works with any number of layers and any neuron count per layer
- ✅ **Accurate Power Modeling**: Sequential MAC design for PrimeTime energy analysis
- ✅ **Fixed-Point Arithmetic**: Q16 format for hardware efficiency
- ✅ **Full Parameterization**: Configure network topology via parameters, not code changes

**Example Configuration (Sample):**
- Layer 1 (Input): 3 neurons
- Layer 2: 5 neurons
- Layer 3: 3 neurons
- Layer 4 (Output): 1 neuron

**But equally works for:** Any topology like [784, 128, 64, 10] (MNIST), [256, 256, 256, 1], [512, 128, 32, 8, 4], etc.

---

## Architecture Overview

## Architecture Overview

### Hierarchical Design (N Layers, M Neurons Per Layer):

```
┌─────────────────────────────────────────────┐
│   MLP Top-Level (mlp_network.sv)           │
│  Parameterized for any network topology    │
│  [N_LAYERS][NEURONS_PER_LAYER]             │
└────────────┬────────────────────────────────┘
             │
    ┌────────┴────────┬────────────┬─────────┬─────┐
    ▼                 ▼            ▼         ▼     ▼
┌─────────┐   ┌─────────────┐ ┌────────┐ ... ┌──────────┐
│Layer 1  │   │  Layer 2    │ │ Layer 3│     │ Layer N  │
│(M1 neu) │   │  (M2 neu)   │ │(M3 neu)│     │ (Mn neu) │
└────┬────┘   └──────┬──────┘ └───┬────┘     └────┬─────┘
     │                │           │              │
     └────────────────┴───────────┴──────────────┘
                   ▲
                   │
            ┌──────────────────┐
            │  neuron_mac_unit │
            │  (Sequential MAC) │
            │  (Core Building  │
            │   Block - DONE)  │
            └──────────────────┘
```

**Features:**
- Network topology fully parameterized
- Automatic interconnection of layers
- Each layer has arbitrary neuron count
- Scales from shallow (2-3 layers) to deep networks

### Neuron Computation:

Each neuron computes a weighted sum plus bias using **sequential MAC (Multiply-Accumulate)**:

$$a_l^n = \sum_{i=0}^{m-1} a_{l-1}^i \cdot w_l^{n,i} + b_l^n$$

Where:
- $a_{l-1}^i$ = activation from previous layer input $i$
- $w_l^{n,i}$ = weight connecting input $i$ to neuron $n$ in layer $l$
- $b_l^n$ = bias for neuron $n$ in layer $l$
- $m$ = number of inputs to neuron

---

## Project Components

### ✅ Completed

#### 1. **neuron_dot_product.sv** - Core MAC Unit
Sequential Multiply-Accumulate module with:
- **Architecture**: One 16×16 multiplier + one adder per cycle
- **Data Format**: Fixed-point Q16 (8.8 bits: 8 integer + 8 fractional)
- **Accumulator**: 48-bit (prevents overflow for large INPUT_WIDTH)
- **Latency**: INPUT_WIDTH + 1 cycles (includes bias addition)
- **Key Features**:
  - Per-product Q32→Q16 shifting (correct numerical Q-format)
  - Guarded array access (no out-of-bounds)
  - Power-accurate for PrimeTime analysis
  - Synthesizable to real hardware (DSP-friendly)

**Parameterizable by:**
- `INPUT_WIDTH`: Number of inputs
- `DATA_WIDTH`: Fixed-point word width (default 16 bits)
- `ACC_WIDTH`: Accumulator width (default 48 bits)

**Example Usage:**
```systemverilog
neuron_dot_product #(.INPUT_WIDTH(3), .DATA_WIDTH(16), .ACC_WIDTH(48))
neuron_instance (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(activations),      // Q16 fixed-point
    .w_in(weights),          // Q16 fixed-point
    .bias(bias_value),       // Q16 fixed-point
    .valid_in(compute_en),
    .a_out(neuron_output),   // Q16 fixed-point
    .valid_out(result_valid)
);
```

#### 2. **tb_neuron_dot_product.sv** - Testbench
Comprehensive testbench with 5 test cases:
1. **Test 1**: Layer 1→2 (INPUT_WIDTH=3)
2. **Test 2**: Layer 2→3 (INPUT_WIDTH=5)
3. **Test 3**: Layer 3→4 (INPUT_WIDTH=1)
4. **Test 4**: Negative values and mixed signs
5. **Test 5**: Zero inputs with non-zero bias

Each test verifies correct Q16 arithmetic and timing.

---

### 🔄 In Progress / To-Do

#### 3. **layer.sv** - Layer Controller
*To be implemented*
- Instantiates N neurons with same inputs
- Distributes activations to all neurons in parallel
- Collects outputs from neurons
- Handles sequential MAC scheduling (pipeline different neurons)
- Timing: ~(INPUT_WIDTH + 1) × NUM_NEURONS cycles (no parallelism)

#### 4. **mlp_network.sv** - Top-Level Network Controller
*To be implemented*
- Instantiates all N layers (configurable via parameter)
- Routes outputs of layer i to inputs of layer i+1
- Manages inference flow and timing across entire network
- Input/output buffering/formatting
- Status signals (inference_done, error, layer_active)

#### 5. **activation_function.sv** - Activation Functions
*To be implemented*
- ReLU: $f(x) = \max(0, x)$
- Sigmoid (optional): $f(x) = \frac{1}{1 + e^{-x}}$
- Linear (for output layer)
- Fixed-point friendly implementations

#### 6. **weight_memory.sv** - Weight Storage
*To be implemented*
- ROM for storing pre-trained weights and biases
- Per-layer organized
- Initialized from hex/binary files or Python scripts

#### 7. **system_controller.sv** - Main FSM & Orchestrator
*To be implemented*
- Orchestrates full inference pipeline across N layers
- Handles layer-by-layer computation sequencing
- Applies parameterized activation functions between layers
- Manages I/O interfaces for input/output vectors
- Timing synchronization across all layers

---

## Generalization & Scalability

### Design is Fully Parameterized:

| Aspect | Configurable | Range |
|--------|--------------|-------|
| Number of Layers | Yes | 1 to N (limited by synthesis) |
| Neurons per Layer | Yes (per layer) | 1 to M (depends on mem/area budget) |
| Input Width (Layer 1) | Yes | 1 to 1024 (or more) |
| Output Width (Layer N) | Automatic | Derived from topology |
| Data Width (Q format) | Yes | 8, 16, 24, 32 bits |
| Accumulator Width | Yes | Must accommodate max layer MAC width |

### Examples of Generalization:

**Narrow & Deep:**
```
Input: 128 → 64 → 32 → 16 → 8 → 4 → 2 (Compression network)
```

**Wide & Shallow:**
```
Input: 784 → 512 → 10 (Simple MNIST)
```

**Variable Widths:**
```
Input: 100 → 200 → 150 → 100 → 50 (Custom architecture)
```

**Pure Convolutional (via flattening):**
```
Input: 3072 (32×32×3) → 256 → 128 → 10 (CNN flattened)
```

---

### Representation:
- **16-bit signed**: range $[-128, 128)$ with $2^{-8} = 1/256$ resolution
- **Conversion**: 
  - Float → Q16: `q16_value = float_value × 256`
  - Q16 → Float: `float_value = q16_value / 256`

### Arithmetic in Module:
- **Multiply**: $Q16 \times Q16 = Q32$ (32-bit result)
- **Per-product shift**: $Q32 >> 8 = Q16$ (before accumulation)
- **Accumulate**: $Q16 + Q16 = Q16$ (in 48-bit accumulator)
- **Final shift**: $Q32 >> 8 = Q16$ (when producing output)

### Example:
```systemverilog
// Input: 0.5 in floating-point
a_in = 128;   // 0.5 × 256

// Input: 0.25 in floating-point
w_in = 64;    // 0.25 × 256

// Product: (128 × 64) = 8192 (Q32)
// Shift: 8192 >> 8 = 32 (Q16, which represents 0.125)
// Expected: 0.5 × 0.25 = 0.125 ✓
```

---

## Testing & Simulation

### Unit Testing (Neuron Module):

```bash
# Using Questa/ModelSim
vsim -do "run -all" tb_neuron_dot_product

# Or compile and simulate
vlog neuron_dot_product.sv tb_neuron_dot_product.sv
vsim tb_neuron_dot_product
```

### Integration Testing (Full Network):

*To be implemented with various topologies:*

```bash
# Test 3-neuron input, 5-neuron hidden, etc.
vsim -g "NETWORK_LAYERS=4" \
     -g "NEURONS_PER_LAYER='[3,5,3,1]" \
     tb_mlp_network

# Test MNIST-sized network
vsim -g "NETWORK_LAYERS=3" \
     -g "NEURONS_PER_LAYER='[784,128,10]" \
     tb_mlp_network

# Test custom configuration
vsim -g "NETWORK_LAYERS=5" \
     -g "NEURONS_PER_LAYER='[256,256,256,128,10]" \
     tb_mlp_network
```

### Expected Output:
```
========================================
  Neuron Sequential MAC Testbench
  Fixed-Point Q16 (8.8) Format
========================================

--- TEST CASE 1: Layer 1->2 (INPUT_WIDTH=3) ---
Q16 Fixed-Point (8.8 format)
Sequential MAC: Load a[0]*w[0], then accumulate a[1]*w[1] and a[2]*w[2]
Inputs (Q16): a_in=[128, 128, 128]
              w_in=[64, 64, 64]
              bias=64
Expected (Q16): ~160 (0.625 in FP)
Output after 4 cycles: [result] (Q16)
✓ Output valid
```

---

## Power Analysis (PrimeTime)

### Key Features for Accurate Energy Modeling:
1. **Sequential MAC Architecture**: One multiplier + one adder toggle per cycle
2. **Input-Dependent Switching**: Power ∝ actual computation work
3. **Per-Product Shifting**: Reduces intermediate widths → less switching
4. **Clear Cycle Attribution**: Easy to map power to specific operations

### PrimeTime Setup:
1. Synthesize `neuron_dot_product.sv` to netlist
2. Back-annotate switching activity from simulation
3. Run power analysis with:
   - Different input patterns
   - Different layer widths
   - Different batch sizes (if pipelined)
4. Extract metrics:
   - Energy per MAC operation
   - Energy per neuron
   - Energy per inference

---

## File Structure

```
MLP-in-Verilog/
├── README.md                          (This file - design overview)
├── neuron_dot_product.sv              ✅ Sequential MAC core unit
├── tb_neuron_dot_product.sv           ✅ Testbench for neuron
├── layer.sv                           🔄 Generic layer controller
├── mlp_network.sv                     🔄 Network topology generator
├── activation_function.sv             🔄 Parameterized activations
├── weight_memory.sv                   🔄 Weight/bias storage (any size)
├── system_controller.sv               🔄 Full inference orchestrator
│
├── samples/                           (Example configurations)
│   ├── config_3_5_3_1.sv             (3→5→3→1 sample)
│   ├── config_784_128_10.sv          (MNIST-like sample)
│   └── config_custom.sv              (User example)
│
└── sim_results/                       (Simulation outputs)
    ├── waves_3_5_3_1/
    ├── waves_784_128_10/
    └── power_analysis/
```

---

## Configuration Parameters

### Network Topology (User-Configurable):

The MLP is fully parameterized. Define network topology as an array:

```systemverilog
// Example 1: [3, 5, 3, 1] - The sample configuration
localparam int NETWORK_LAYERS = 4;
localparam int NEURONS_PER_LAYER[NETWORK_LAYERS] = '{3, 5, 3, 1};

// Example 2: [784, 128, 64, 10] - MNIST classifier
localparam int NETWORK_LAYERS = 4;
localparam int NEURONS_PER_LAYER[NETWORK_LAYERS] = '{784, 128, 64, 10};

// Example 3: [256, 256, 256, 1] - Shallow fully-connected
localparam int NETWORK_LAYERS = 4;
localparam int NEURONS_PER_LAYER[NETWORK_LAYERS] = '{256, 256, 256, 1};

// Example 4: [512, 256, 128, 64, 32, 8, 4, 2] - Deep network
localparam int NETWORK_LAYERS = 8;
localparam int NEURONS_PER_LAYER[NETWORK_LAYERS] = '{512, 256, 128, 64, 32, 8, 4, 2};
```

### Fixed-Point Settings (Hardware Tuning):

```systemverilog
localparam DATA_WIDTH = 16;     // Q16 (adjust for precision: 8, 16, 24, 32)
localparam ACC_WIDTH = 48;      // Accumulator (adjust for max layer size)
localparam Q_SHIFT = 8;         // Fractional bits (must match DATA_WIDTH layout)
```

### Layer Architecture (Auto-Derived):

```systemverilog
// Automatically computed from NEURONS_PER_LAYER
localparam int LAYER_INPUT_WIDTH[NETWORK_LAYERS-1:0];  // Input size to each layer
localparam int LAYER_OUTPUT_WIDTH[NETWORK_LAYERS:0];   // Output size of each layer
localparam int MAX_NEURONS = max(NEURONS_PER_LAYER);   // Largest layer
localparam int MAX_WEIGHTS = MAX_NEURONS * MAX_INPUT;  // Largest MAC width
```

---

## Design Principles

1. **Synthesizability**: No unsynthesizable constructs (no `real`, full `logic` types)
2. **Power Accuracy**: Sequential MACs for input-dependent switching
3. **Scalability**: Parameterized by layer width and network depth
4. **Clarity**: Explicit dataflow and state machines
5. **Testability**: Comprehensive testbenches for each module

---

## Next Steps

1. **Implement `layer.sv`**: Generalized layer controller for N neurons with shared inputs
2. **Implement `mlp_network.sv`**: Automatic layer instantiation and interconnection
3. **Add activation functions**: Parameterized ReLU, Sigmoid, Linear, Tanh
4. **Integrate weight/bias memory**: ROM with configurable depth/width for any topology
5. **Write system testbench**: Full inference test with various topologies (3-5-3-1, 784-128-10, etc.)
6. **Synthesize & Power Analysis**: Run PrimeTime on complete design with multiple configs
7. **Document scalability results**: Show area, timing, power vs. network depth and width

---

## References & Tools

- **Simulation**: Questa/ModelSim, VCS, Vivado
- **Synthesis**: Synopsys Design Compiler, Vivado, Quartus
- **Power Analysis**: PrimeTime PX, Joules
- **Fixed-Point Arithmetic**: Q (Qm.n) format standard

---

## Author & License

Capstone Project: MLP Hardware Accelerator in SystemVerilog  
Date: February 2026

