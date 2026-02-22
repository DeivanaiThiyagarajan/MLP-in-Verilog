`timescale 1ns / 1ps

// Testbench for Three Interconnected Neurons
// Architecture: 
//   - Neuron 1: 3 inputs + 3 weights -> output1
//   - Neuron 2: 3 inputs + 3 weights -> output2
//   - Neuron 3: output1 & output2 + 2 weights -> final_output
// All neurons use ReLU activation

module tb_three_neurons_interconnected();

    // ==================== PARAMETERS ====================
    localparam NEURON_WIDTH_L1 = 3;          // Width for layer 1 neurons (3 inputs each)
    localparam NEURON_WIDTH_L2 = 2;          // Width for layer 2 neuron (2 inputs from L1)
    localparam NEURON_BITS = 15;             // Data width (16 bits = 1 sign + 15 bits)
    localparam COUNTER_END_L1 = NEURON_WIDTH_L1 - 1;  // 0 to 2
    localparam COUNTER_END_L2 = NEURON_WIDTH_L2 - 1;  // 0 to 1
    localparam B_BITS = 15;                  // Bias bits
    localparam CLK_PERIOD = 10;              // 10ns clock period (100MHz)
    
    // ==================== SIGNALS ====================
    
    // Clock and Reset
    reg clk;
    reg rstn;
    
    // ========== LAYER 1: Two neurons in parallel ==========
    
    // Neuron 1 (Layer 1)
    reg signed [31:0] weights_n1 [0:NEURON_WIDTH_L1];
    reg signed [NEURON_BITS:0] data_in_n1 [0:NEURON_WIDTH_L1];
    reg signed [B_BITS:0] bias_n1;
    reg [31:0] counter_n1_out;
    reg counter_n1_done;
    reg signed [NEURON_BITS + 8:0] neuron_n1_out;
    
    // Neuron 2 (Layer 1)
    reg signed [31:0] weights_n2 [0:NEURON_WIDTH_L1];
    reg signed [NEURON_BITS:0] data_in_n2 [0:NEURON_WIDTH_L1];
    reg signed [B_BITS:0] bias_n2;
    reg [31:0] counter_n2_out;
    reg counter_n2_done;
    reg signed [NEURON_BITS + 8:0] neuron_n2_out;
    
    // ========== LAYER 2: One neuron (takes N1 and N2 outputs) ==========
    
    // Neuron 3 (Layer 2) - takes outputs from N1 and N2 as inputs
    reg signed [31:0] weights_n3 [0:NEURON_WIDTH_L2];
    reg signed [NEURON_BITS:0] data_in_n3 [0:NEURON_WIDTH_L2];
    reg signed [B_BITS:0] bias_n3;
    reg [31:0] counter_n3_out;
    reg counter_n3_done;
    reg signed [NEURON_BITS + 8:0] neuron_n3_out;
    
    // Variables for verification
    reg [63:0] expected_sum;
    reg [63:0] expected_output;
    reg [63:0] n1_expected;
    reg [63:0] n2_expected;
    reg [31:0] test_count;
    reg [31:0] pass_count;
    
    // ==================== DUT INSTANTIATION ====================
    
    // ========== LAYER 1 COUNTERS ==========
    counter #(.END_COUNTER(COUNTER_END_L1)) counter_n1_inst (
        .clk(clk),
        .rstn(rstn),
        .counter_out(counter_n1_out),
        .counter_donestatus(counter_n1_done)
    );
    
    counter #(.END_COUNTER(COUNTER_END_L1)) counter_n2_inst (
        .clk(clk),
        .rstn(rstn),
        .counter_out(counter_n2_out),
        .counter_donestatus(counter_n2_done)
    );
    
    // ========== LAYER 2 COUNTER ==========
    counter #(.END_COUNTER(COUNTER_END_L2)) counter_n3_inst (
        .clk(clk),
        .rstn(rstn),
        .counter_out(counter_n3_out),
        .counter_donestatus(counter_n3_done)
    );
    
    // ========== LAYER 1 NEURONS ==========
    neuron_inputlayer #(
        .NEURON_WIDTH(NEURON_WIDTH_L1),
        .NEURON_BITS(NEURON_BITS),
        .COUNTER_END(COUNTER_END_L1),
        .B_BITS(B_BITS)
    ) neuron_1_dut (
        .clk(clk),
        .rstn(rstn),
        .activation_function(1'b1),      // Enable ReLU
        .weights(weights_n1),
        .data_in(data_in_n1),
        .b(bias_n1),
        .counter(counter_n1_out),
        .data_out(neuron_n1_out)
    );
    
    neuron_inputlayer #(
        .NEURON_WIDTH(NEURON_WIDTH_L1),
        .NEURON_BITS(NEURON_BITS),
        .COUNTER_END(COUNTER_END_L1),
        .B_BITS(B_BITS)
    ) neuron_2_dut (
        .clk(clk),
        .rstn(rstn),
        .activation_function(1'b1),      // Enable ReLU
        .weights(weights_n2),
        .data_in(data_in_n2),
        .b(bias_n2),
        .counter(counter_n2_out),
        .data_out(neuron_n2_out)
    );
    
    // ========== LAYER 2 NEURON ==========
    neuron_inputlayer #(
        .NEURON_WIDTH(NEURON_WIDTH_L2),
        .NEURON_BITS(NEURON_BITS),
        .COUNTER_END(COUNTER_END_L2),
        .B_BITS(B_BITS)
    ) neuron_3_dut (
        .clk(clk),
        .rstn(rstn),
        .activation_function(1'b1),      // Enable ReLU
        .weights(weights_n3),
        .data_in(data_in_n3),
        .b(bias_n3),
        .counter(counter_n3_out),
        .data_out(neuron_n3_out)
    );
    
    // ==================== CLOCK GENERATION ====================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ==================== TEST PROCEDURE ====================
    initial begin
        
        $dumpfile("three_neurons_sim.vcd");
        $dumpvars(0, tb_three_neurons_interconnected);

        $display("\n");
        $display("====================================================================");
        $display("  THREE INTERCONNECTED NEURONS TESTBENCH");
        $display("  Architecture: 2-Layer Network");
        $display("  Layer 1: Neuron_1 (3 inputs), Neuron_2 (3 inputs)");
        $display("  Layer 2: Neuron_3 (takes N1 & N2 outputs as inputs)");
        $display("  Activation: ReLU on all neurons");
        $display("====================================================================\n");
        
        // Initialize signals
        test_count = 0;
        pass_count = 0;
        rstn = 0;
        
        // Reset release
        #(2 * CLK_PERIOD);
        rstn = 1;
        
        #(CLK_PERIOD);
        
        // ========== TEST CASE 1: Simple Positive Values ==========
        test_case_1();
        
        // ========== TEST CASE 2: Mixed Positive and Negative ==========
        test_case_2();
        
        // ========== TEST CASE 3: Large Values (Stress Test) ==========
        test_case_3();
        
        // ========== TEST CASE 4: Negative Values in Layer 1 ==========
        //test_case_4();
        
        #(10 * CLK_PERIOD);
        
        $display("\n");
        $display("====================================================================");
        $display("  TEST SUMMARY");
        $display("  Total Tests: %0d", test_count);
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", test_count - pass_count);
        $display("====================================================================\n");
        
        $finish;
    end

    // ==================== HELPER TASK: Run Layer 1 Neurons ==========
    task run_layer1();
        $display("\n  [Layer 1] Computing outputs from Neuron_1 and Neuron_2...");
        
        // Reset counters
        rstn = 0;
        #(CLK_PERIOD);
        rstn = 1;
        #(CLK_PERIOD);
        
        // Wait for both neurons to complete
        $display("  Waiting for Neuron_1 and Neuron_2 computation...");
        wait(counter_n1_done == 1'b1);
        wait(counter_n2_done == 1'b1);
        $display("  Layer 1 computation completed at t=%0t ns", $time);
        
        // Wait for output propagation
        repeat(NEURON_WIDTH_L1+2) @(posedge clk);
        
        $display("  Neuron_1 Output: %0d", neuron_n1_out);
        $display("  Neuron_2 Output: %0d", neuron_n2_out);
    endtask

    // ==================== HELPER TASK: Run Layer 2 Neuron ==========
    task run_layer2();
        $display("\n  [Layer 2] Computing output from Neuron_3...");
        $display("  Input to Neuron_3: neuron_1_out=%0d, neuron_2_out=%0d", neuron_n1_out, neuron_n2_out);
        
        // Feed Layer 1 outputs as Layer 2 inputs
        data_in_n3[0] = neuron_n1_out[NEURON_BITS:0];
        data_in_n3[1] = neuron_n2_out[NEURON_BITS:0];
        
        // Reset counter for Layer 2
        rstn = 0;
        #(CLK_PERIOD);
        rstn = 1;
        #(CLK_PERIOD);
        
        // Wait for Neuron_3 to complete
        $display("  Waiting for Neuron_3 computation...");
        wait(counter_n3_done == 1'b1);
        $display("  Layer 2 computation completed at t=%0t ns", $time);
        
        // Wait for output propagation
        repeat(NEURON_WIDTH_L2+3) @(posedge clk);
        
        $display("  Neuron_3 Output (Final): %0d", neuron_n3_out);
    endtask

    // ========== VERIFICATION HELPER ==========
    task verify_result(input reg [63:0] expected, input reg [63:0] actual, input reg [255:0] test_name);
        test_count = test_count + 1;
        
        if (actual == expected) begin
            $display("  ✓ PASS: %s", test_name);
            $display("    Expected: %0d, Got: %0d\n", expected, actual);
            pass_count = pass_count + 1;
        end else begin
            $display("  ✗ FAIL: %s", test_name);
            $display("    Expected: %0d, Got: %0d", expected, actual);
            $display("    Difference: %0d\n", actual - expected);
        end
    endtask

    // ==================== TEST CASE 1 ====================
    task test_case_1();
        $display("\n");
        $display("===== TEST CASE 1: Simple Positive Values =====");
        $display("----------------------------------------------");
        
        // Configure Layer 1 neurons
        // Neuron 1: inputs=[2,3,4], weights=[5,6,7], bias=10
        data_in_n1[0] = 16'sd2;
        data_in_n1[1] = 16'sd3;
        data_in_n1[2] = 16'sd4;
        weights_n1[0] = 32'sd5;
        weights_n1[1] = 32'sd6;
        weights_n1[2] = 32'sd7;
        bias_n1 = 16'sd10;
        
        // Neuron 2: inputs=[1,2,3], weights=[4,5,6], bias=5
        data_in_n2[0] = 16'sd1;
        data_in_n2[1] = 16'sd2;
        data_in_n2[2] = 16'sd3;
        weights_n2[0] = 32'sd4;
        weights_n2[1] = 32'sd5;
        weights_n2[2] = 32'sd6;
        bias_n2 = 16'sd5;
        
        // Display configurations
        $display("\n  Layer 1 Configuration:");
        $display("  Neuron_1 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n1[0], data_in_n1[1], data_in_n1[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n1[0], weights_n1[1], weights_n1[2]);
        $display("    Bias:    %0d", bias_n1);
        $display("  Neuron_2 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n2[0], data_in_n2[1], data_in_n2[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n2[0], weights_n2[1], weights_n2[2]);
        $display("    Bias:    %0d", bias_n2);
        
        // Manual calculation for Layer 1
        expected_sum = (2*5) + (3*6) + (4*7) + 10;
        $display("\n  Neuron_1 Manual Calc: (2*5) + (3*6) + (4*7) + 10 = %0d", expected_sum);
        expected_output = expected_sum;  // Positive, so no ReLU clipping
        
        $display("  Neuron_1 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        n1_expected = expected_output;
        
        expected_sum = (1*4) + (2*5) + (3*6) + 5;
        $display("  Neuron_2 Manual Calc: (1*4) + (2*5) + (3*6) + 5 = %0d", expected_sum);
        expected_output = expected_sum;  // Positive, so no ReLU clipping
        
        $display("  Neuron_2 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        n2_expected = expected_output;
        
        // Run Layer 1
        run_layer1();
        verify_result(n1_expected, neuron_n1_out, "Neuron_1 Output");
        verify_result(n2_expected, neuron_n2_out, "Neuron_2 Output");
        
        // Configure Layer 2 neuron - uses outputs from Layer 1
        weights_n3[0] = 32'sd3;
        weights_n3[1] = 32'sd2;
        bias_n3 = 16'sd5;
        
        $display("  Layer 2 Configuration:");
        $display("  Neuron_3 (Layer 2):");
        $display("    Will take Neuron_1 and Neuron_2 outputs as inputs");
        $display("    Weights: [%0d, %0d]", weights_n3[0], weights_n3[1]);
        $display("    Bias: %0d", bias_n3);
        
        // Manual calculation for Layer 2
        expected_sum = (n1_expected * 3) + (n2_expected * 2) + 5;
        $display("\n  Neuron_3 Manual Calc: (%0d*3) + (%0d*2) + 5 = %0d", n1_expected, n2_expected, expected_sum);
        expected_output = expected_sum;  // Positive, so no ReLU clipping
        $display("  Neuron_3 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        // Run Layer 2
        run_layer2();
        verify_result(expected_output, neuron_n3_out, "Neuron_3 Final Output");
        
        // Reset for next test
        rstn = 0;
        repeat(10) @(posedge clk);
        rstn = 1;
    endtask

    // ==================== TEST CASE 2 ====================
    task test_case_2();
        $display("\n");
        $display("===== TEST CASE 2: Mixed Positive and Negative =====");
        $display("---------------------------------------------------");
        
        // Configure Layer 1 neurons with mixed values
        // Neuron 1: inputs=[-2,5,-1], weights=[3,2,8], bias=5
        data_in_n1[0] = -16'sd2;
        data_in_n1[1] = 16'sd5;
        data_in_n1[2] = -16'sd1;
        weights_n1[0] = 32'sd3;
        weights_n1[1] = 32'sd2;
        weights_n1[2] = 32'sd8;
        bias_n1 = 16'sd5;
        
        // Neuron 2: inputs=[3,-2,4], weights=[2,3,1], bias=2
        data_in_n2[0] = 16'sd3;
        data_in_n2[1] = -16'sd2;
        data_in_n2[2] = 16'sd4;
        weights_n2[0] = 32'sd2;
        weights_n2[1] = 32'sd3;
        weights_n2[2] = 32'sd1;
        bias_n2 = 16'sd2;
        
        // Display configurations
        $display("\n  Layer 1 Configuration:");
        $display("  Neuron_1 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n1[0], data_in_n1[1], data_in_n1[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n1[0], weights_n1[1], weights_n1[2]);
        $display("    Bias:    %0d", bias_n1);
        $display("  Neuron_2 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n2[0], data_in_n2[1], data_in_n2[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n2[0], weights_n2[1], weights_n2[2]);
        $display("    Bias:    %0d", bias_n2);
        
        // Manual calculation for Layer 1
        expected_sum = (-2*3) + (5*2) + (-1*8) + 5;
        $display("\n  Neuron_1 Manual Calc: (-2*3) + (5*2) + (-1*8) + 5 = %0d", expected_sum);
        expected_output = (expected_sum < 0) ? 0 : expected_sum;
        $display("  Neuron_1 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        n1_expected = expected_output;
        
        expected_sum = (3*2) + (-2*3) + (4*1) + 2;
        $display("  Neuron_2 Manual Calc: (3*2) + (-2*3) + (4*1) + 2 = %0d", expected_sum);
        expected_output = (expected_sum < 0) ? 0 : expected_sum;
        $display("  Neuron_2 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        n2_expected = expected_output;
        
        // Run Layer 1
        run_layer1();
        verify_result(n1_expected, neuron_n1_out, "Neuron_1 Output");
        verify_result(n2_expected, neuron_n2_out, "Neuron_2 Output");
        
        // Configure Layer 2 neuron
        weights_n3[0] = 32'sd2;
        weights_n3[1] = 32'sd4;
        bias_n3 = 16'sd3;
        
        $display("  Layer 2 Configuration:");
        $display("  Neuron_3 (Layer 2):");
        $display("    Will take Neuron_1 and Neuron_2 outputs as inputs");
        $display("    Weights: [%0d, %0d]", weights_n3[0], weights_n3[1]);
        $display("    Bias: %0d", bias_n3);
        
        // Manual calculation for Layer 2
        expected_sum = (n1_expected * 2) + (n2_expected * 4) + 3;
        $display("\n  Neuron_3 Manual Calc: (%0d*2) + (%0d*4) + 3 = %0d", n1_expected, n2_expected, expected_sum);
        expected_output = (expected_sum < 0) ? 0 : expected_sum;
        $display("  Neuron_3 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        // Run Layer 2
        run_layer2();
        verify_result(expected_output, neuron_n3_out, "Neuron_3 Final Output");
        
        // Reset for next test
        rstn = 0;
        repeat(10) @(posedge clk);
        rstn = 1;
    endtask

    // ==================== TEST CASE 3 ====================
    task test_case_3();
        $display("\n");
        $display("===== TEST CASE 3: Large Values (Stress Test) =====");
        $display("--------------------------------------------------");
        
        // Configure Layer 1 neurons with large values
        // Neuron 1: inputs=[50,100,75], weights=[10,20,15], bias=50
        data_in_n1[0] = 16'sd50;
        data_in_n1[1] = 16'sd100;
        data_in_n1[2] = 16'sd75;
        weights_n1[0] = 32'sd10;
        weights_n1[1] = 32'sd20;
        weights_n1[2] = 32'sd15;
        bias_n1 = 16'sd50;
        
        // Neuron 2: inputs=[30,60,40], weights=[5,8,12], bias=25
        data_in_n2[0] = 16'sd30;
        data_in_n2[1] = 16'sd60;
        data_in_n2[2] = 16'sd40;
        weights_n2[0] = 32'sd5;
        weights_n2[1] = 32'sd8;
        weights_n2[2] = 32'sd12;
        bias_n2 = 16'sd25;
        
        // Display configurations
        $display("\n  Layer 1 Configuration:");
        $display("  Neuron_1 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n1[0], data_in_n1[1], data_in_n1[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n1[0], weights_n1[1], weights_n1[2]);
        $display("    Bias:    %0d", bias_n1);
        $display("  Neuron_2 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n2[0], data_in_n2[1], data_in_n2[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n2[0], weights_n2[1], weights_n2[2]);
        $display("    Bias:    %0d", bias_n2);
        
        // Manual calculation for Layer 1
        expected_sum = (50*10) + (100*20) + (75*15) + 50;
        $display("\n  Neuron_1 Manual Calc: (50*10) + (100*20) + (75*15) + 50 = %0d", expected_sum);
        expected_output = expected_sum;
        $display("  Neuron_1 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        n1_expected = expected_output;
        
        expected_sum = (30*5) + (60*8) + (40*12) + 25;
        $display("  Neuron_2 Manual Calc: (30*5) + (60*8) + (40*12) + 25 = %0d", expected_sum);
        expected_output = expected_sum;
        $display("  Neuron_2 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        n2_expected = expected_output;
        
        // Run Layer 1
        run_layer1();
        verify_result(n1_expected, neuron_n1_out, "Neuron_1 Output");
        verify_result(n2_expected, neuron_n2_out, "Neuron_2 Output");
        
        // Configure Layer 2 neuron
        weights_n3[0] = 32'sd5;
        weights_n3[1] = 32'sd3;
        bias_n3 = 16'sd100;
        
        $display("  Layer 2 Configuration:");
        $display("  Neuron_3 (Layer 2):");
        $display("    Will take Neuron_1 and Neuron_2 outputs as inputs");
        $display("    Weights: [%0d, %0d]", weights_n3[0], weights_n3[1]);
        $display("    Bias: %0d", bias_n3);
        
        // Manual calculation for Layer 2
        expected_sum = (n1_expected * 5) + (n2_expected * 3) + 100;
        $display("\n  Neuron_3 Manual Calc: (%0d*5) + (%0d*3) + 100 = %0d", n1_expected, n2_expected, expected_sum);
        expected_output = expected_sum;
        $display("  Neuron_3 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        // Run Layer 2
        run_layer2();
        verify_result(expected_output, neuron_n3_out, "Neuron_3 Final Output");
        
        // Reset for next test
        rstn = 0;
        repeat(10) @(posedge clk);
        rstn = 1;
    endtask

    // ==================== TEST CASE 4 ====================
    task test_case_4();
        $display("\n");
        $display("===== TEST CASE 4: Negative Values in Layer 1 =====");
        $display("---------------------------------------------------");
        
        // Configure Layer 1 neurons where outputs become 0 due to ReLU
        // Neuron 1: inputs=[-5,-3,-2], weights=[2,3,4], bias=-50 (will result in negative sum)
        data_in_n1[0] = -16'sd5;
        data_in_n1[1] = -16'sd3;
        data_in_n1[2] = -16'sd2;
        weights_n1[0] = 32'sd2;
        weights_n1[1] = 32'sd3;
        weights_n1[2] = 32'sd4;
        bias_n1 = -16'sd50;
        
        // Neuron 2: inputs=[2,3,1], weights=[5,6,7], bias=20
        data_in_n2[0] = 16'sd2;
        data_in_n2[1] = 16'sd3;
        data_in_n2[2] = 16'sd1;
        weights_n2[0] = 32'sd5;
        weights_n2[1] = 32'sd6;
        weights_n2[2] = 32'sd7;
        bias_n2 = 16'sd20;
        
        // Display configurations
        $display("\n  Layer 1 Configuration:");
        $display("  Neuron_1 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n1[0], data_in_n1[1], data_in_n1[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n1[0], weights_n1[1], weights_n1[2]);
        $display("    Bias:    %0d", bias_n1);
        $display("  Neuron_2 Configuration:");
        $display("    Inputs:  [%0d, %0d, %0d]", data_in_n2[0], data_in_n2[1], data_in_n2[2]);
        $display("    Weights: [%0d, %0d, %0d]", weights_n2[0], weights_n2[1], weights_n2[2]);
        $display("    Bias:    %0d", bias_n2);
        
        // Manual calculation for Layer 1
        expected_sum = (-5*2) + (-3*3) + (-2*4) + (-50);
        $display("\n  Neuron_1 Manual Calc: (-5*2) + (-3*3) + (-2*4) + (-50) = %0d", expected_sum);
        expected_output = (expected_sum < 0) ? 0 : expected_sum;
        $display("  Neuron_1 Expected Output (ReLU): max(0, %0d) = %0d (Clipped to 0)", expected_sum, expected_output);
        
        n1_expected = expected_output;
        
        expected_sum = (2*5) + (3*6) + (1*7) + 20;
        $display("  Neuron_2 Manual Calc: (2*5) + (3*6) + (1*7) + 20 = %0d", expected_sum);
        expected_output = expected_sum;
        $display("  Neuron_2 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        n2_expected = expected_output;
        
        // Run Layer 1
        run_layer1();
        verify_result(n1_expected, neuron_n1_out, "Neuron_1 Output (Clipped to 0)");
        verify_result(n2_expected, neuron_n2_out, "Neuron_2 Output");
        
        // Configure Layer 2 neuron
        weights_n3[0] = 32'sd10;
        weights_n3[1] = 32'sd5;
        bias_n3 = 16'sd50;
        
        $display("  Layer 2 Configuration:");
        $display("  Neuron_3 (Layer 2):");
        $display("    Will take Neuron_1 and Neuron_2 outputs as inputs");
        $display("    Weights: [%0d, %0d]", weights_n3[0], weights_n3[1]);
        $display("    Bias: %0d", bias_n3);
        
        // Manual calculation for Layer 2
        expected_sum = (n1_expected * 10) + (n2_expected * 5) + 50;
        $display("\n  Neuron_3 Manual Calc: (%0d*10) + (%0d*5) + 50 = %0d", n1_expected, n2_expected, expected_sum);
        expected_output = expected_sum;
        $display("  Neuron_3 Expected Output (ReLU): max(0, %0d) = %0d", expected_sum, expected_output);
        
        // Run Layer 2
        run_layer2();
        verify_result(expected_output, neuron_n3_out, "Neuron_3 Final Output");
        
        // Reset for next test
        rstn = 0;
        repeat(10) @(posedge clk);
        rstn = 1;
    endtask
    
endmodule
