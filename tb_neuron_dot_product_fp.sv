// Testbench for neuron_dot_product_fp module (Fixed-Point Sequential MAC)
// Tests the module with Q8.8 fixed-point format (8 integer bits + 8 fractional bits)
// Note: Sequential MAC takes INPUT_WIDTH + 1 cycles per computation
// Q8.8 Format: To convert decimal to Q8.8, multiply by 256 (2^8)
// Example: 0.5 in Q8.8 = 0.5 * 256 = 128
// Multiply: Q8.8 * Q8.8 = Q16.16 (32-bit), shift right 8 bits to get Q8.8

`timescale 1ns/1ps

module tb_neuron_dot_product_fp();

    localparam INPUT_WIDTH_TEST1 = 3;
    localparam INPUT_WIDTH_TEST2 = 5;
    localparam INPUT_WIDTH_TEST3 = 2;
    localparam FRAC_BITS = 8;
    
    logic clk, rst_n;
    
    // Test signals for test case 1 (INPUT_WIDTH=3)
    logic signed [15:0] a_in_1 [INPUT_WIDTH_TEST1-1:0];
    logic signed [15:0] w_in_1 [INPUT_WIDTH_TEST1-1:0];
    logic signed [15:0] bias_1;
    logic valid_in_1, valid_out_1;
    logic signed [15:0] a_out_1;
    
    // Test signals for test case 2 (INPUT_WIDTH=5)
    logic signed [15:0] a_in_2 [INPUT_WIDTH_TEST2-1:0];
    logic signed [15:0] w_in_2 [INPUT_WIDTH_TEST2-1:0];
    logic signed [15:0] bias_2;
    logic valid_in_2, valid_out_2;
    logic signed [15:0] a_out_2;
    
    // Test signals for test case 3 (INPUT_WIDTH=2)
    logic signed [15:0] a_in_3 [INPUT_WIDTH_TEST3-1:0];
    logic signed [15:0] w_in_3 [INPUT_WIDTH_TEST3-1:0];
    logic signed [15:0] bias_3;
    logic valid_in_3, valid_out_3;
    logic signed [15:0] a_out_3;
    
    // DUT instantiations
    neuron_dot_product_fp #(.INPUT_WIDTH(INPUT_WIDTH_TEST1), .FRAC_BITS(FRAC_BITS)) dut1 (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_in_1), .w_in(w_in_1), .bias(bias_1),
        .valid_in(valid_in_1), .valid_out(valid_out_1), .a_out(a_out_1));
    
    neuron_dot_product_fp #(.INPUT_WIDTH(INPUT_WIDTH_TEST2), .FRAC_BITS(FRAC_BITS)) dut2 (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_in_2), .w_in(w_in_2), .bias(bias_2),
        .valid_in(valid_in_2), .valid_out(valid_out_2), .a_out(a_out_2));
          
    neuron_dot_product_fp #(.INPUT_WIDTH(INPUT_WIDTH_TEST3), .FRAC_BITS(FRAC_BITS)) dut3 (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_in_3), .w_in(w_in_3), .bias(bias_3),
        .valid_in(valid_in_3), .valid_out(valid_out_3), .a_out(a_out_3));
    
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("========================================");
        $display("  Neuron Sequential MAC Testbench");
        $display("  Fixed-Point Q8.8 Format");
        $display("========================================\n");
        
        rst_n = 1'b0;
        valid_in_1 = 1'b0;
        valid_in_2 = 1'b0;
        valid_in_3 = 1'b0;
        
        #20 rst_n = 1'b1;
        
        // TEST CASE 1: INPUT_WIDTH = 3
        $display("--- TEST CASE 1: Layer 1->2 (INPUT_WIDTH=3) ---");
        
        a_in_1[0] = 128;      // 0.5 in Q8.8
        a_in_1[1] = 77;       // 0.3 in Q8.8
        a_in_1[2] = 51;       // 0.2 in Q8.8
        
        w_in_1[0] = 205;      // 0.8 in Q8.8
        w_in_1[1] = 154;      // 0.6 in Q8.8
        w_in_1[2] = 102;      // 0.4 in Q8.8
        
        bias_1 = 26;          // 0.1 in Q8.8
        valid_in_1 = 1'b1;
        
        $display("Inputs: a=[128, 77, 51], w=[205, 154, 102], bias=26 (Q8.8 format)");
        $display("Calculations (using floor division for negative arithmetic right shift):");
        $display("  Product 1: (128 * 205) >> 8 = 26240 >> 8 = 102 (102.5 truncated towards zero)");
        $display("  Product 2: (77 * 154) >> 8 = 11858 >> 8 = 46 (46.3 truncated)");
        $display("  Product 3: (51 * 102) >> 8 = 5202 >> 8 = 20 (20.3 truncated)");
        $display("  Sum: 102 + 46 + 20 = 168");
        $display("  Expected Result: 168 + 26 = 194");
        
        @(posedge clk);
        valid_in_1 = 1'b0;
        
        wait(valid_out_1 == 1'b1);
        #1;
        
        $display("Output: %d, valid_out=%b\n", a_out_1, valid_out_1);
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST CASE 2: INPUT_WIDTH = 5
        $display("--- TEST CASE 2: Layer 2->3 (INPUT_WIDTH=5) ---");
        
        a_in_2[0] = 26;       // 0.1 in Q8.8
        a_in_2[1] = 51;       // 0.2 in Q8.8
        a_in_2[2] = 77;       // 0.3 in Q8.8
        a_in_2[3] = 102;      // 0.4 in Q8.8
        a_in_2[4] = 128;      // 0.5 in Q8.8
        
        w_in_2[0] = 128;      // 0.5 in Q8.8
        w_in_2[1] = 102;      // 0.4 in Q8.8
        w_in_2[2] = 77;       // 0.3 in Q8.8
        w_in_2[3] = 51;       // 0.2 in Q8.8
        w_in_2[4] = 26;       // 0.1 in Q8.8
        
        bias_2 = 64;          // 0.25 in Q8.8
        valid_in_2 = 1'b1;
        
        $display("Inputs: a=[26,51,77,102,128], w=[128,102,77,51,26], bias=64 (Q8.8 format)");
        $display("Calculations:");
        $display("  Product 1: (26 * 128) >> 8 = 3328 >> 8 = 13");
        $display("  Product 2: (51 * 102) >> 8 = 5202 >> 8 = 20");
        $display("  Product 3: (77 * 77) >> 8 = 5929 >> 8 = 23");
        $display("  Product 4: (102 * 51) >> 8 = 5202 >> 8 = 20");
        $display("  Product 5: (128 * 26) >> 8 = 3328 >> 8 = 13");
        $display("  Sum: 13 + 20 + 23 + 20 + 13 = 89");
        $display("  Expected Result: 89 + 64 = 153");
        
        @(posedge clk);
        valid_in_2 = 1'b0;
        
        wait(valid_out_2 == 1'b1);
        #1;
        
        $display("Output: %d, valid_out=%b\n", a_out_2, valid_out_2);
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST CASE 3: INPUT_WIDTH = 2
        $display("--- TEST CASE 3: 2-Input Neuron (INPUT_WIDTH=2) ---");
        
        a_in_3[0] = 384;      // 1.5 in Q8.8
        a_in_3[1] = 640;      // 2.5 in Q8.8
        
        w_in_3[0] = 77;       // 0.3 in Q8.8
        w_in_3[1] = 179;      // 0.7 in Q8.8
        
        bias_3 = 128;         // 0.5 in Q8.8
        valid_in_3 = 1'b1;
        
        $display("Inputs: a=[384, 640], w=[77, 179], bias=128 (Q8.8 format)");
        $display("Calculations:");
        $display("  Product 1: (384 * 77) >> 8 = 29568 >> 8 = 115");
        $display("  Product 2: (640 * 179) >> 8 = 114560 >> 8 = 447");
        $display("  Sum: 115 + 447 = 562");
        $display("  Expected Result: 562 + 128 = 690");
        
        @(posedge clk);
        valid_in_3 = 1'b0;
        
        wait(valid_out_3 == 1'b1);
        #1;
        
        $display("Output: %d, valid_out=%b\n", a_out_3, valid_out_3);
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST CASE 4: Negative values
        $display("--- TEST CASE 4: Negative inputs/weights (INPUT_WIDTH=3) ---");
        
        a_in_1[0] = -128;     // -0.5 in Q8.8
        a_in_1[1] = 77;       // 0.3 in Q8.8
        a_in_1[2] = -51;      // -0.2 in Q8.8
        
        w_in_1[0] = 102;      // 0.4 in Q8.8
        w_in_1[1] = 51;       // 0.2 in Q8.8
        w_in_1[2] = 77;       // 0.3 in Q8.8
        
        bias_1 = 0;
        valid_in_1 = 1'b1;
        
        $display("Inputs: a=[-128, 77, -51], w=[102, 51, 77], bias=0 (Q8.8 format)");
        $display("Calculations (using floor division for negative arithmetic right shift):");
        $display("  Product 1: (-128 * 102) >> 8 = -13056 >> 8 = -51 (exact)");
        $display("  Product 2: (77 * 51) >> 8 = 3927 >> 8 = 15 (15.34 truncated)");
        $display("  Product 3: (-51 * 77) >> 8 = -3927 >> 8 = -16 (floor of -15.34)");
        $display("  Sum: -51 + 15 + (-16) = -52");
        $display("  Expected Result: -52 + 0 = -52");
        
        @(posedge clk);
        valid_in_1 = 1'b0;
        
        wait(valid_out_1 == 1'b1);
        #1;
        
        $display("Output: %d, valid_out=%b\n", a_out_1, valid_out_1);
        
        // Reset modules
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST CASE 5: Zero inputs
        $display("--- TEST CASE 5: Zero inputs (INPUT_WIDTH=3) ---");
        
        a_in_1[0] = 0;
        a_in_1[1] = 0;
        a_in_1[2] = 0;
        
        w_in_1[0] = 128;      // 0.5 in Q8.8
        w_in_1[1] = 154;      // 0.6 in Q8.8
        w_in_1[2] = 179;      // 0.7 in Q8.8
        
        bias_1 = 384;         // 1.5 in Q8.8
        valid_in_1 = 1'b1;
        
        $display("Inputs: a=[0, 0, 0], w=[128, 154, 179], bias=384 (Q8.8 format)");
        $display("Calculations:");
        $display("  Product 1: (0 * 128) >> 8 = 0");
        $display("  Product 2: (0 * 154) >> 8 = 0");
        $display("  Product 3: (0 * 179) >> 8 = 0");
        $display("  Sum: 0 + 0 + 0 = 0");
        $display("  Expected Result: 0 + 384 = 384");
        
        @(posedge clk);
        valid_in_1 = 1'b0;
        
        wait(valid_out_1 == 1'b1);
        #1;
        
        $display("Output: %d, valid_out=%b\n", a_out_1, valid_out_1);
        
        $display("========================================");
        $display("  Testbench Complete");
        $display("  Fixed-Point Q8.8 Sequential MAC");
        $display("========================================");
        
        $finish;
    end

endmodule
