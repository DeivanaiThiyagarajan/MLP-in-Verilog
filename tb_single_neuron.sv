`timescale 1ns / 1ps

// Testbench for Single Neuron
// Tests one neuron with specific inputs, weights, and bias

module tb_single_neuron();

    // ==================== PARAMETERS ====================
    localparam NEURON_WIDTH = 10;        // Number of inputs to neuron
    localparam NEURON_BITS = 15;        // Data width (16 bits = 1 sign + 15 bits)
    localparam COUNTER_END = NEURON_WIDTH - 1;  // Counter: 0 to NEURON_WIDTH-1 (3 iterations)
    localparam B_BITS = 15;             // Bias bits
    localparam CLK_PERIOD = 10;         // 10ns clock period (100MHz)
    
    // ==================== SIGNALS ====================
    
    // Clock and Reset
    reg clk;
    reg rstn;
    
    // Neuron signals
    reg signed [31:0] weights [0:NEURON_WIDTH];
    reg signed [NEURON_BITS:0] data_in [0:NEURON_WIDTH];
    reg signed [B_BITS:0] bias;
    reg [31:0] counter_out;
    reg counter_done;
    reg signed [NEURON_BITS + 8:0] neuron_out;
    
    // Variables for verification
    reg [63:0] expected_sum;
    reg [63:0] expected_output;
    
    // ==================== DUT INSTANTIATION ====================
    
    // Counter module
    counter #(.END_COUNTER(COUNTER_END)) counter_inst (
        .clk(clk),
        .rstn(rstn),
        .counter_out(counter_out),
        .counter_donestatus(counter_done)
    );
    
    // Single Neuron under test
    neuron_inputlayer #(
        .NEURON_WIDTH(NEURON_WIDTH),
        .NEURON_BITS(NEURON_BITS),
        .COUNTER_END(COUNTER_END),
        .B_BITS(B_BITS)
    ) neuron_dut (
        .clk(clk),
        .rstn(rstn),
        .activation_function(1'b1),      // Enable ReLU
        .weights(weights),
        .data_in(data_in),
        .b(bias),
        .counter(counter_out),
        .data_out(neuron_out)
    );
    
    // ==================== CLOCK GENERATION ====================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    // ==================== TEST PROCEDURE ====================
    initial begin

        $display("Counter_out after clock generation =%0d", counter_out);
        $display("Counter_done after clock generation =%0d", counter_done);
    
        $dumpfile("single_neuron_sim.vcd");
        $dumpvars(0, tb_single_neuron);

        $display("Counter_out before initialize reset =%0d", counter_out);
        $display("Counter_done before initialize reset =%0d", counter_done);
    
        // Initialize
        rstn = 0;

        $display("Counter_out after initialize reset =%0d", counter_out);
        $display("Counter_done after initialize reset =%0d", counter_done);
    
        
        $display("\n");
        $display("================================================");
        $display("  Single Neuron Testbench");
        $display("  Testing neuron_inputlayer.sv");
        $display("  Architecture: Sequential MAC with ReLU");
        $display("================================================\n");
        
        $display("Counter_out after initial displays =%0d", counter_out);
        $display("Counter_done after initial displays =%0d", counter_done);
    
        // Reset release
        #(2 * CLK_PERIOD);

        $display("Counter_out after 2 clk period =%0d", counter_out);
        $display("Counter_done after 2 clk period =%0d", counter_done);
    
        rstn = 1;

        $display("Counter_out after reset 1 =%0d", counter_out);
        $display("Counter_done after reset 1 =%0d", counter_done);
    
        #(CLK_PERIOD);

        $display("Counter_out after 1 clk period =%0d", counter_out);
        $display("Counter_done after 1 clk period =%0d", counter_done);
    

        test_case_2();
        
        // ========== TEST CASE 1: Simple Positive Values ==========
        test_case_1();
        
        
        // ========== TEST CASE 3: Large Values ==========
        test_case_3();
        
        #(10 * CLK_PERIOD);
        $finish;
    end

    // ==================== TEST CASE 2 ====================
    task test_case_2();
        $display("\n");
        $display("========== TEST CASE 2: Mixed Positive and Negative ==========");
        $display("-------------------------------------------------------------");
        
        // Setup inputs
        data_in[0] = -16'sd2;
        data_in[1] = 16'sd5;
        data_in[2] = -16'sd1;
        data_in[3] = 16'sd10;
        data_in[4] = 16'sd3;
        data_in[5] = -16'sd4;
        data_in[6] = 16'sd7;
        data_in[7] = -16'sd6;
        data_in[8] = 16'sd2;
        data_in[9] = 16'sd8;

        weights[0] = 32'sd3;
        weights[1] = 32'sd2;
        weights[2] = 32'sd8;
        weights[3] = 32'sd10;
        weights[4] = 32'sd1;
        weights[5] = 32'sd2;
        weights[6] = 32'sd4;
        weights[7] = 32'sd3;
        weights[8] = 32'sd5;
        weights[9] = 32'sd2;

        bias = 16'sd5;

        // Display setup
        $display("Input Data:    a[0]=%0d, a[1]=%0d, a[2]=%0d, a[3]=%0d, a[4]=%0d, a[5]=%0d, a[6]=%0d, a[7]=%0d, a[8]=%0d, a[9]=%0d",
                data_in[0], data_in[1], data_in[2], data_in[3], data_in[4], data_in[5], data_in[6], data_in[7], data_in[8], data_in[9]);

        $display("Weights:       w[0]=%0d, w[1]=%0d, w[2]=%0d, w[3]=%0d, w[4]=%0d, w[5]=%0d, w[6]=%0d, w[7]=%0d, w[8]=%0d, w[9]=%0d",
                weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7], weights[8], weights[9]);

        $display("Bias:          b=%0d", bias);

        // Manual calculation
        expected_sum = (-2*3) + (5*2) + (-1*8) + (10*10) + (3*1) + (-4*2) + (7*4) + (-6*3) + (2*5) + (8*2) + 5;

        $display("Manual Calc:   (-2*3) + (5*2) + (-1*8) + (10*10) + (3*1) + (-4*2) + (7*4) + (-6*3) + (2*5) + (8*2) + 5");
        $display("               = %0d", expected_sum);

        if (expected_sum < 0) begin
            expected_output = 0;  // ReLU clamps negative to 0
            $display("ReLU Output:   max(0, %0d) = 0", expected_sum);
        end else begin
            expected_output = expected_sum;
            $display("ReLU Output:   max(0, %0d) = %0d", expected_sum, expected_sum);
        end
        
        // Reset counter
        rstn = 0;
        #(CLK_PERIOD);
        rstn = 1;
        #(CLK_PERIOD);
        
        // Wait for counter to complete
        $display("\nWaiting for computation...");
        wait(counter_done == 1'b1);
        $display("Counter completed at t=%0t ns", $time);
        
        // Wait a few more cycles for output propagation
        repeat(5) @(posedge clk);
        
        $display("Neuron Output at t=%0t ns: %0d", $time, neuron_out);
        
        // Verification
        $display("\n--- VERIFICATION ---");
        if (neuron_out == expected_output) begin
            $display("✓ PASS: Test Case 2");
            $display("  Expected: %0d", expected_output);
            $display("  Got:      %0d", neuron_out);
        end else begin
            $display("✗ FAIL: Test Case 2");
            $display("  Expected: %0d", expected_output);
            $display("  Got:      %0d", neuron_out);
            $display("  Difference: %0d", neuron_out - expected_output);
        end
        rstn = 0;
        // Wait before next test - allow all signals to settle
        repeat(NEURON_WIDTH*3) @(posedge clk);
        rstn = 1;
    endtask

    
    // ==================== TEST CASE 1 ====================
    task test_case_1();
        $display("\n");
        $display("========== TEST CASE 1: Simple Positive Values ==========");
        $display("---------------------------------------------------------");
        
        // Setup inputs
        data_in[0] = 16'sd2;
        data_in[1] = 16'sd3;
        data_in[2] = 16'sd4;
        data_in[3] = 16'sd1;
        data_in[4] = 16'sd5;
        data_in[5] = 16'sd2;
        data_in[6] = 16'sd3;
        data_in[7] = 16'sd4;
        data_in[8] = 16'sd1;
        data_in[9] = 16'sd2;

        weights[0] = 32'sd5;
        weights[1] = 32'sd6;
        weights[2] = 32'sd7;
        weights[3] = 32'sd2;
        weights[4] = 32'sd3;
        weights[5] = 32'sd4;
        weights[6] = 32'sd5;
        weights[7] = 32'sd6;
        weights[8] = 32'sd1;
        weights[9] = 32'sd2;

        bias = 16'sd10;

        // Display setup
        $display("Input Data:    a[0]=%0d, a[1]=%0d, a[2]=%0d, a[3]=%0d, a[4]=%0d, a[5]=%0d, a[6]=%0d, a[7]=%0d, a[8]=%0d, a[9]=%0d",
                data_in[0], data_in[1], data_in[2], data_in[3], data_in[4], data_in[5], data_in[6], data_in[7], data_in[8], data_in[9]);

        $display("Weights:       w[0]=%0d, w[1]=%0d, w[2]=%0d, w[3]=%0d, w[4]=%0d, w[5]=%0d, w[6]=%0d, w[7]=%0d, w[8]=%0d, w[9]=%0d",
                weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7], weights[8], weights[9]);

        $display("Bias:          b=%0d", bias);

        // Manual calculation
        expected_sum = (2*5) + (3*6) + (4*7) + (1*2) + (5*3) + (2*4) + (3*5) + (4*6) + (1*1) + (2*2) + 10;

        $display("Manual Calc:   (2*5) + (3*6) + (4*7) + (1*2) + (5*3) + (2*4) + (3*5) + (4*6) + (1*1) + (2*2) + 10");
        $display("               = %0d", expected_sum);
        
        if (expected_sum < 0) begin
            expected_output = 0;  // ReLU clamps negative to 0
            $display("ReLU Output:   max(0, %0d) = 0", expected_sum);
        end else begin
            expected_output = expected_sum;
            $display("ReLU Output:   max(0, %0d) = %0d", expected_sum, expected_sum);
        end
        
        // Reset counter
        rstn = 0;
        #(CLK_PERIOD);
        rstn = 1;
        #(CLK_PERIOD);
        
        // Wait for counter to complete
        $display("\nWaiting for computation...");
        wait(counter_done == 1'b1);
        $display("Counter completed at t=%0t ns", $time);
        
        // Wait a few more cycles for output propagation
        repeat(5) @(posedge clk);
        
        $display("Neuron Output at t=%0t ns: %0d", $time, neuron_out);
        
        // Verification
        $display("\n--- VERIFICATION ---");
        if (neuron_out == expected_output) begin
            $display("✓ PASS: Test Case 1");
            $display("  Expected: %0d", expected_output);
            $display("  Got:      %0d", neuron_out);
        end else begin
            $display("✗ FAIL: Test Case 1");
            $display("  Expected: %0d", expected_output);
            $display("  Got:      %0d", neuron_out);
            $display("  Difference: %0d", neuron_out - expected_output);
        end
        
        rstn = 0;
        // Wait before next test - allow all signals to settle
        repeat(NEURON_WIDTH*3) @(posedge clk);
        rstn = 1;
    endtask
        
    // ==================== TEST CASE 3 ====================
    task test_case_3();
        $display("\n");
        $display("========== TEST CASE 3: Large Values (Stress Test) ==========");
        $display("-----------------------------------------------------------");
        
        // Setup inputs with larger values
        data_in[0] = 16'sd50;
        data_in[1] = 16'sd100;
        data_in[2] = 16'sd75;
        data_in[3] = 16'sd25;
        data_in[4] = 16'sd60;
        data_in[5] = 16'sd80;
        data_in[6] = 16'sd90;
        data_in[7] = 16'sd40;
        data_in[8] = 16'sd30;
        data_in[9] = 16'sd70;

        weights[0] = 32'sd10;
        weights[1] = 32'sd20;
        weights[2] = 32'sd15;
        weights[3] = 32'sd5;
        weights[4] = 32'sd12;
        weights[5] = 32'sd18;
        weights[6] = 32'sd25;
        weights[7] = 32'sd8;
        weights[8] = 32'sd7;
        weights[9] = 32'sd14;

        bias = 16'sd50;

        // Display setup
        $display("Input Data:    a[0]=%0d, a[1]=%0d, a[2]=%0d, a[3]=%0d, a[4]=%0d, a[5]=%0d, a[6]=%0d, a[7]=%0d, a[8]=%0d, a[9]=%0d",
                data_in[0], data_in[1], data_in[2], data_in[3], data_in[4], data_in[5], data_in[6], data_in[7], data_in[8], data_in[9]);

        $display("Weights:       w[0]=%0d, w[1]=%0d, w[2]=%0d, w[3]=%0d, w[4]=%0d, w[5]=%0d, w[6]=%0d, w[7]=%0d, w[8]=%0d, w[9]=%0d",
                weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7], weights[8], weights[9]);

        $display("Bias:          b=%0d", bias);

        // Manual calculation
        expected_sum = (50*10) + (100*20) + (75*15) + (25*5) + (60*12) + (80*18) + (90*25) + (40*8) + (30*7) + (70*14) + 50;

        $display("Manual Calc:   (50*10) + (100*20) + (75*15) + (25*5) + (60*12) + (80*18) + (90*25) + (40*8) + (30*7) + (70*14) + 50");
        $display("               = %0d", expected_sum);

        if (expected_sum < 0) begin
            expected_output = 0;  // ReLU clamps negative to 0
            $display("ReLU Output:   max(0, %0d) = 0", expected_sum);
        end else begin
            expected_output = expected_sum;
            $display("ReLU Output:   max(0, %0d) = %0d", expected_sum, expected_sum);
        end
        
        // Reset counter
        rstn = 0;
        #(CLK_PERIOD);
        rstn = 1;
        #(CLK_PERIOD);
        
        // Wait for counter to complete
        $display("\nWaiting for computation...");
        wait(counter_done == 1'b1);
        $display("Counter completed at t=%0t ns", $time);
        
        // Wait a few more cycles for output propagation
        repeat(5) @(posedge clk);
        
        $display("Neuron Output at t=%0t ns: %0d", $time, neuron_out);
        
        // Verification
        $display("\n--- VERIFICATION ---");
        if (neuron_out == expected_output) begin
            $display("✓ PASS: Test Case 3");
            $display("  Expected: %0d", expected_output);
            $display("  Got:      %0d", neuron_out);
        end else begin
            $display("✗ FAIL: Test Case 3");
            $display("  Expected: %0d", expected_output);
            $display("  Got:      %0d", neuron_out);
            $display("  Difference: %0d", neuron_out - expected_output);
        end
        
        // Wait before next test - allow all signals to settle
        repeat(NEURON_WIDTH*3) @(posedge clk);
    endtask
    
endmodule
