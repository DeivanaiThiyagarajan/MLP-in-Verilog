// Testbench for ReLU module
// Tests ReLU function f(a) = max(0, a) for various input configurations

`timescale 1ns/1ps

module tb_relu();

    localparam INPUT_WIDTH = 3;
    localparam DATA_WIDTH = 16;
    
    logic signed [DATA_WIDTH-1:0] a_in [INPUT_WIDTH-1:0];
    logic signed [DATA_WIDTH-1:0] a_out [INPUT_WIDTH-1:0];
    
    // DUT instantiation
    relu #(.INPUT_WIDTH(INPUT_WIDTH), .DATA_WIDTH(DATA_WIDTH)) dut (
        .a_in(a_in),
        .a_out(a_out)
    );
    
    initial begin
        $display("=====================================");
        $display("  ReLU Activation Function Test");
        $display("  f(a) = max(0, a)");
        $display("=====================================\n");
        
        // TEST CASE 1: Positive values
        $display("--- TEST CASE 1: All positive values ---");
        a_in[0] = 10;
        a_in[1] = 25;
        a_in[2] = 50;
        
        #1;
        
        $display("Input:  a=[10, 25, 50]");
        $display("Expected: [10, 25, 50] (positive values pass through)");
        $display("Output:   a=[%d, %d, %d]\n", a_out[0], a_out[1], a_out[2]);
        
        // TEST CASE 2: Negative values
        $display("--- TEST CASE 2: All negative values ---");
        a_in[0] = -10;
        a_in[1] = -25;
        a_in[2] = -50;
        
        #1;
        
        $display("Input:  a=[-10, -25, -50]");
        $display("Expected: [0, 0, 0] (negative values become 0)");
        $display("Output:   a=[%d, %d, %d]\n", a_out[0], a_out[1], a_out[2]);
        
        // TEST CASE 3: Mixed positive and negative
        $display("--- TEST CASE 3: Mixed positive and negative ---");
        a_in[0] = 15;
        a_in[1] = -20;
        a_in[2] = 30;
        
        #1;
        
        $display("Input:  a=[15, -20, 30]");
        $display("Expected: [15, 0, 30]");
        $display("Output:   a=[%d, %d, %d]\n", a_out[0], a_out[1], a_out[2]);
        
        // TEST CASE 4: Zero values
        $display("--- TEST CASE 4: Zero values ---");
        a_in[0] = 0;
        a_in[1] = 0;
        a_in[2] = 0;
        
        #1;
        
        $display("Input:  a=[0, 0, 0]");
        $display("Expected: [0, 0, 0] (zero remains zero)");
        $display("Output:   a=[%d, %d, %d]\n", a_out[0], a_out[1], a_out[2]);
        
        // TEST CASE 5: Mixed with zero
        $display("--- TEST CASE 5: Mixed with zero ---");
        a_in[0] = -100;
        a_in[1] = 0;
        a_in[2] = 100;
        
        #1;
        
        $display("Input:  a=[-100, 0, 100]");
        $display("Expected: [0, 0, 100]");
        $display("Output:   a=[%d, %d, %d]\n", a_out[0], a_out[1], a_out[2]);
        
        // TEST CASE 6: Edge case - min/max values
        $display("--- TEST CASE 6: Min and max values ---");
        a_in[0] = -32768;  // Most negative for 16-bit signed
        a_in[1] = 0;
        a_in[2] = 32767;   // Most positive for 16-bit signed
        
        #1;
        
        $display("Input:  a=[-32768, 0, 32767]");
        $display("Expected: [0, 0, 32767]");
        $display("Output:   a=[%d, %d, %d]\n", a_out[0], a_out[1], a_out[2]);
        
        // TEST CASE 7: Different input widths (INPUT_WIDTH=5)
        $display("--- TEST CASE 7: Verification at INPUT_WIDTH=5 ---");
        $display("Testing with 5 inputs to verify parameterizable width");
        a_in[0] = 5;
        a_in[1] = -3;
        if (INPUT_WIDTH > 2) a_in[2] = 8;
        if (INPUT_WIDTH > 3) a_in[3] = -15;
        if (INPUT_WIDTH > 4) a_in[4] = 12;
        
        #1;
        
        if (INPUT_WIDTH == 3) begin
            $display("Note: Current INPUT_WIDTH=3, test case 7 applies ReLU to first 3 elements");
            $display("Input:  a=[5, -3, 8]");
            $display("Expected: [5, 0, 8]");
            $display("Output:   a=[%d, %d, %d]\n", a_out[0], a_out[1], a_out[2]);
        end else begin
            $display("Input:  a=[5, -3, 8, -15, 12]");
            $display("Expected: [5, 0, 8, 0, 12]");
            $display("Output:   a=[%d, %d, %d, %d, %d]\n", a_out[0], a_out[1], a_out[2], a_out[3], a_out[4]);
        end
        
        $display("=====================================");
        $display("  ReLU Tests Complete");
        $display("=====================================");
        
        $finish;
    end

endmodule
