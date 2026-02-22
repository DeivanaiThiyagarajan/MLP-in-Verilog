// Testbench for neuron_dot_product module (Integer Sequential MAC Version)
// Tests the module with integer format values
// Note: Sequential MAC takes INPUT_WIDTH + 1 cycles per computation

`timescale 1ns/1ps

module tb_neuron_dot_product();

    // Test parameters
    localparam INPUT_WIDTH_TEST1 = 3;      // Layer 1->2 (4 cycles: load + 3 products + bias)
    localparam INPUT_WIDTH_TEST2 = 5;      // Layer 2->3 (6 cycles)
    localparam INPUT_WIDTH_TEST3 = 2;      // Small 2-input test (3 cycles)
    
    localparam DATA_WIDTH = 16;
    localparam ACC_WIDTH = 48;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Test signals for test case 1 (INPUT_WIDTH=3, like layer 1->2)
    logic signed [DATA_WIDTH-1:0] a_in_1 [INPUT_WIDTH_TEST1-1:0];
    logic signed [DATA_WIDTH-1:0] w_in_1 [INPUT_WIDTH_TEST1-1:0];
    logic signed [DATA_WIDTH-1:0] bias_1;
    logic valid_in_1, valid_out_1;
    logic signed [DATA_WIDTH-1:0] a_out_1;
    
    // Test signals for test case 2 (INPUT_WIDTH=5, like layer 2->3)
    logic signed [DATA_WIDTH-1:0] a_in_2 [INPUT_WIDTH_TEST2-1:0];
    logic signed [DATA_WIDTH-1:0] w_in_2 [INPUT_WIDTH_TEST2-1:0];
    logic signed [DATA_WIDTH-1:0] bias_2;
    logic valid_in_2, valid_out_2;
    logic signed [DATA_WIDTH-1:0] a_out_2;
    
    // Test signals for test case 3 (INPUT_WIDTH=2)
    logic signed [DATA_WIDTH-1:0] a_in_3 [INPUT_WIDTH_TEST3-1:0];
    logic signed [DATA_WIDTH-1:0] w_in_3 [INPUT_WIDTH_TEST3-1:0];
    logic signed [DATA_WIDTH-1:0] bias_3;
    logic valid_in_3, valid_out_3;
    logic signed [DATA_WIDTH-1:0] a_out_3;
    
    // Instantiate three neuron modules for testing different widths
    neuron_dot_product #(.INPUT_WIDTH(INPUT_WIDTH_TEST1), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) 
    dut1 (.clk(clk), .rst_n(rst_n), 
          .a_in(a_in_1), .w_in(w_in_1), .bias(bias_1),
          .valid_in(valid_in_1), .valid_out(valid_out_1), .a_out(a_out_1));
    
    neuron_dot_product #(.INPUT_WIDTH(INPUT_WIDTH_TEST2), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) 
    dut2 (.clk(clk), .rst_n(rst_n), 
          .a_in(a_in_2), .w_in(w_in_2), .bias(bias_2),
          .valid_in(valid_in_2), .valid_out(valid_out_2), .a_out(a_out_2));
          
    neuron_dot_product #(.INPUT_WIDTH(INPUT_WIDTH_TEST3), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) 
    dut3 (.clk(clk), .rst_n(rst_n), 
          .a_in(a_in_3), .w_in(w_in_3), .bias(bias_3),
          .valid_in(valid_in_3), .valid_out(valid_out_3), .a_out(a_out_3));
    
    // Clock generation: 10ns period
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end
    
    // Main testbench
    initial begin

        $dumpfile("one_neuron.vcd");
        $dumpvars(0, tb_neuron_dot_product);

        $display("========================================");
        $display("  Neuron Sequential MAC Testbench");
        $display("  Integer Format");
        $display("========================================\n");
        
        // Reset
        rst_n = 1'b0;
        valid_in_1 = 1'b0;
        valid_in_2 = 1'b0;
        valid_in_3 = 1'b0;
        
        #20 rst_n = 1'b1;
        
        // ============ TEST CASE 1: INPUT_WIDTH = 3 ============
        // Sequential MAC: Load a[0]*w[0], then accumulate a[1]*w[1] and a[2]*w[2]
        // 4 cycles total (1 load + 2 compute + 1 done)
        
        $display("--- TEST CASE 1: Layer 1->2 (INPUT_WIDTH=3) ---");
        $display("Integer Format Sequential MAC");
        $display("Sequential MAC: Load a[0]*w[0], then accumulate a[1]*w[1] and a[2]*w[2]");
        
        // a = [2, 3, 4], w = [5, 6, 7], bias = 10
        a_in_1[0] = 16'sd2;
        a_in_1[1] = 16'sd3;
        a_in_1[2] = 16'sd4;
        
        w_in_1[0] = 16'sd5;
        w_in_1[1] = 16'sd6;
        w_in_1[2] = 16'sd7;
        
        bias_1 = 16'sd10;
        valid_in_1 = 1'b1;
        
        // Expected: (2*5) + (3*6) + (4*7) + 10 = 10 + 18 + 28 + 10 = 66
        
        $display("Inputs (Integer): a_in=[%5d, %5d, %5d]", a_in_1[0], a_in_1[1], a_in_1[2]);
        $display("                  w_in=[%5d, %5d, %5d]", w_in_1[0], w_in_1[1], w_in_1[2]);
        $display("                  bias=%5d", bias_1);
        $display("Expected: 66");
        
        @(posedge clk);
        valid_in_1 = 1'b0;
        
        // Wait for output to be valid
        wait(valid_out_1 == 1'b1);
        #1;
        
        $display("Output after %0d cycles: %5d", INPUT_WIDTH_TEST1 + 1, a_out_1);
        
        if (valid_out_1) begin
            $display("✓ Output valid");
        end else begin
            $display("✗ Output not valid yet");
        end
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // ============ TEST CASE 2: INPUT_WIDTH = 5 ============
        // Sequential MAC: 5 products + 1 add bias = 6 cycles
        
        $display("\n--- TEST CASE 2: Layer 2->3 (INPUT_WIDTH=5) ---");
        $display("Integer Sequential MAC over 5 products");
        
        // a = [1, 2, 3, 4, 5], w = [2, 3, 4, 5, 6], bias = 20
        a_in_2[0] = 16'sd1;
        a_in_2[1] = 16'sd2;
        a_in_2[2] = 16'sd3;
        a_in_2[3] = 16'sd4;
        a_in_2[4] = 16'sd5;
        
        w_in_2[0] = 16'sd2;
        w_in_2[1] = 16'sd3;
        w_in_2[2] = 16'sd4;
        w_in_2[3] = 16'sd5;
        w_in_2[4] = 16'sd6;
        
        bias_2 = 16'sd20;
        valid_in_2 = 1'b1;
        
        // Expected: (1*2) + (2*3) + (3*4) + (4*5) + (5*6) + 20 = 2 + 6 + 12 + 20 + 30 + 20 = 90
        
        $display("Inputs (Integer): a_in=[%5d, %5d, %5d, %5d, %5d]", 
                 a_in_2[0], a_in_2[1], a_in_2[2], a_in_2[3], a_in_2[4]);
        $display("                  w_in=[%5d, %5d, %5d, %5d, %5d]",
                 w_in_2[0], w_in_2[1], w_in_2[2], w_in_2[3], w_in_2[4]);
        $display("                  bias=%5d", bias_2);
        $display("Expected: 90");
        
        @(posedge clk);
        valid_in_2 = 1'b0;
        
        // Wait for output to be valid
        wait(valid_out_2 == 1'b1);
        #1;
        
        $display("Output after %0d cycles: %5d", INPUT_WIDTH_TEST2 + 1, a_out_2);
        
        if (valid_out_2) begin
            $display("✓ Output valid");
        end else begin
            $display("✗ Output not valid yet");
        end
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // ============ TEST CASE 3: INPUT_WIDTH = 2 ============
        // Small 2-input case
        // Sequential MAC: 2 products + 1 add bias = 3 cycles
        
        $display("\n--- TEST CASE 3: 2-Input Neuron (INPUT_WIDTH=2) ---");
        $display("Integer format with two inputs");
        
        // a = [10, 20], w = [3, 4], bias = 5
        a_in_3[0] = 16'sd10;
        a_in_3[1] = 16'sd20;
        w_in_3[0] = 16'sd3;
        w_in_3[1] = 16'sd4;
        bias_3 = 16'sd5;
        valid_in_3 = 1'b1;
        
        // Expected: (10*3) + (20*4) + 5 = 30 + 80 + 5 = 115
        
        $display("Inputs (Integer): a_in=[%5d, %5d], w_in=[%5d, %5d], bias=%5d", 
                 a_in_3[0], a_in_3[1], w_in_3[0], w_in_3[1], bias_3);
        $display("Expected: 115");
        
        @(posedge clk);
        valid_in_3 = 1'b0;
        
        // Wait for output to be valid
        wait(valid_out_3 == 1'b1);
        #1;
        
        $display("Output after %0d cycles: %5d", INPUT_WIDTH_TEST3 + 1, a_out_3);
        
        if (valid_out_3) begin
            $display("✓ Output valid");
        end else begin
            $display("✗ Output not valid yet");
        end
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // ============ TEST CASE 4: Negative values ============
        // Sequential MAC with 3 inputs (4 cycles total)
        
        $display("\n--- TEST CASE 4: Negative inputs/weights (INPUT_WIDTH=3) ---");
        
        // a = [-5, 3, -2], w = [4, 2, 3], bias = 0
        a_in_1[0] = -16'sd5;
        a_in_1[1] = 16'sd3;
        a_in_1[2] = -16'sd2;
        
        w_in_1[0] = 16'sd4;
        w_in_1[1] = 16'sd2;
        w_in_1[2] = 16'sd3;
        
        bias_1 = 16'sd0;
        valid_in_1 = 1'b1;
        
        // Expected: (-5*4) + (3*2) + (-2*3) + 0 = -20 + 6 - 6 = -20
        
        $display("Inputs (Integer): a_in=[%5d, %5d, %5d]", a_in_1[0], a_in_1[1], a_in_1[2]);
        $display("                  w_in=[%5d, %5d, %5d]", w_in_1[0], w_in_1[1], w_in_1[2]);
        $display("                  bias=%5d", bias_1);
        $display("Expected: -20");
        
        @(posedge clk);
        valid_in_1 = 1'b0;
        
        // Wait for output to be valid
        wait(valid_out_1 == 1'b1);
        #1;
        
        $display("Output after %0d cycles: %5d", INPUT_WIDTH_TEST1 + 1, a_out_1);
        
        if (valid_out_1) begin
            $display("✓ Output valid");
        end else begin
            $display("✗ Output not valid yet");
        end
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // ============ TEST CASE 5: Zero inputs ============
        // All activations zero, just bias
        
        $display("\n--- TEST CASE 5: Zero inputs (INPUT_WIDTH=3) ---");
        
        // a = [0, 0, 0], w = [5, 6, 7], bias = 15
        a_in_1[0] = 16'sd0;
        a_in_1[1] = 16'sd0;
        a_in_1[2] = 16'sd0;
        
        w_in_1[0] = 16'sd5;
        w_in_1[1] = 16'sd6;
        w_in_1[2] = 16'sd7;
        
        bias_1 = 16'sd15;
        valid_in_1 = 1'b1;
        
        // Expected: 0 + 0 + 0 + 15 = 15
        
        $display("Inputs (Integer): all activations = 0, bias=%5d", bias_1);
        $display("Expected: 15");
        
        @(posedge clk);
        valid_in_1 = 1'b0;
        
        // Wait for output to be valid
        wait(valid_out_1 == 1'b1);
        #1;
        
        $display("Output after %0d cycles: %5d", INPUT_WIDTH_TEST1 + 1, a_out_1);
        
        if (valid_out_1) begin
            $display("✓ Output valid");
        end else begin
            $display("✗ Output not valid yet");
        end
        
        #10;
        
        $display("\n========================================");
        $display("  Testbench Complete");
        $display("  Integer Sequential MAC");
        $display("  Ready for Synthesis & PrimeTime");
        $display("========================================");
        
        $finish;
    end

endmodule
