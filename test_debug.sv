// Debug test for negative and zero inputs with integer format
`timescale 1ns/1ps

module test_debug();
    logic clk, rst_n;
    logic signed [15:0] a_in[2:0];
    logic signed [15:0] w_in[2:0];
    logic signed [15:0] bias;
    logic valid_in, valid_out;
    logic signed [15:0] a_out;
    
    neuron_dot_product #(.INPUT_WIDTH(3)) dut (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_in), .w_in(w_in), .bias(bias),
        .valid_in(valid_in), .valid_out(valid_out), .a_out(a_out)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $display("========================================");
        $display("  Integer Format Debug Tests");
        $display("  Testing Negative and Zero Inputs");
        $display("========================================\n");
        
        rst_n = 0;
        valid_in = 0;
        #20 rst_n = 1;
        
        // TEST 1: ZERO INPUTS with positive weights and bias
        $display("--- TEST 1: ALL ZERO INPUTS ---");
        a_in[0] = 0; a_in[1] = 0; a_in[2] = 0;
        w_in[0] = 5; w_in[1] = 6; w_in[2] = 7;
        bias = 15;
        valid_in = 1;
        $display("Inputs: a=[0, 0, 0], w=[5, 6, 7], bias=15");
        $display("Expected: (0*5) + (0*6) + (0*7) + 15 = 15");
        
        @(posedge clk);
        valid_in = 0;
        
        // Wait for valid_out to go high
        wait(valid_out == 1'b1);
        #1;
        $display("Output: %5d, valid_out=%b\n", a_out, valid_out);
        
        // Reset module state
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST 2: NEGATIVE INPUTS
        $display("--- TEST 2: NEGATIVE INPUTS ---");
        a_in[0] = -5; a_in[1] = 3; a_in[2] = -2;
        w_in[0] = 4; w_in[1] = 2; w_in[2] = 3;
        bias = 0;
        valid_in = 1;
        $display("Inputs: a=[-5, 3, -2], w=[4, 2, 3], bias=0");
        $display("Expected: (-5*4) + (3*2) + (-2*3) + 0 = -20 + 6 - 6 = -20");
        
        @(posedge clk);
        valid_in = 0;
        
        // Wait for valid_out to go high
        wait(valid_out == 1'b1);
        #1;
        $display("Output: %5d, valid_out=%b\n", a_out, valid_out);
        
        // Reset module state
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST 3: ALL NEGATIVE INPUTS
        $display("--- TEST 3: ALL NEGATIVE INPUTS ---");
        a_in[0] = -10; a_in[1] = -20; a_in[2] = -30;
        w_in[0] = -1; w_in[1] = -2; w_in[2] = -3;
        bias = -50;
        valid_in = 1;
        $display("Inputs: a=[-10, -20, -30], w=[-1, -2, -3], bias=-50");
        $display("Expected: (-10*-1) + (-20*-2) + (-30*-3) - 50 = 10 + 40 + 90 - 50 = 90");
        
        @(posedge clk);
        valid_in = 0;
        
        // Wait for valid_out to go high
        wait(valid_out == 1'b1);
        #1;
        $display("Output: %5d, valid_out=%b\n", a_out, valid_out);
        
        // Reset module state
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST 4: MIXED POSITIVE AND NEGATIVE
        $display("--- TEST 4: MIXED POSITIVE/NEGATIVE ---");
        a_in[0] = 10; a_in[1] = -15; a_in[2] = 8;
        w_in[0] = -2; w_in[1] = 3; w_in[2] = -4;
        bias = 100;
        valid_in = 1;
        $display("Inputs: a=[10, -15, 8], w=[-2, 3, -4], bias=100");
        $display("Expected: (10*-2) + (-15*3) + (8*-4) + 100 = -20 - 45 - 32 + 100 = 3");
        
        @(posedge clk);
        valid_in = 0;
        
        // Wait for valid_out to go high
        wait(valid_out == 1'b1);
        #1;
        $display("Output: %5d, valid_out=%b\n", a_out, valid_out);
        
        // Reset module state
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST 5: ZERO BIAS
        $display("--- TEST 5: ZERO BIAS ---");
        a_in[0] = 7; a_in[1] = -3; a_in[2] = 5;
        w_in[0] = 2; w_in[1] = 4; w_in[2] = -1;
        bias = 0;
        valid_in = 1;
        $display("Inputs: a=[7, -3, 5], w=[2, 4, -1], bias=0");
        $display("Expected: (7*2) + (-3*4) + (5*-1) + 0 = 14 - 12 - 5 = -3");
        
        @(posedge clk);
        valid_in = 0;
        
        // Wait for valid_out to go high
        wait(valid_out == 1'b1);
        #1;
        $display("Output: %5d, valid_out=%b\n", a_out, valid_out);
        
        // Reset module state
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST 6: NEGATIVE RESULT
        $display("--- TEST 6: NEGATIVE RESULT ---");
        a_in[0] = 3; a_in[1] = 4; a_in[2] = 5;
        w_in[0] = -2; w_in[1] = -3; w_in[2] = -4;
        bias = 10;
        valid_in = 1;
        $display("Inputs: a=[3, 4, 5], w=[-2, -3, -4], bias=10");
        $display("Expected: (3*-2) + (4*-3) + (5*-4) + 10 = -6 - 12 - 20 + 10 = -28");
        
        @(posedge clk);
        valid_in = 0;
        
        // Wait for valid_out to go high
        wait(valid_out == 1'b1);
        #1;
        $display("Output: %5d, valid_out=%b\n", a_out, valid_out);
        
        // Reset module state
        rst_n = 0;
        #10;
        rst_n = 1;
        #10;
        
        // TEST 7: LARGE NEGATIVE VALUES
        $display("--- TEST 7: LARGE NEGATIVE VALUES ---");
        a_in[0] = -100; a_in[1] = -50; a_in[2] = 30;
        w_in[0] = 5; w_in[1] = 10; w_in[2] = -2;
        bias = 1000;
        valid_in = 1;
        $display("Inputs: a=[-100, -50, 30], w=[5, 10, -2], bias=1000");
        $display("Expected: (-100*5) + (-50*10) + (30*-2) + 1000 = -500 - 500 - 60 + 1000 = -60");
        
        @(posedge clk);
        valid_in = 0;
        
        // Wait for valid_out to go high
        wait(valid_out == 1'b1);
        #1;
        $display("Output: %5d, valid_out=%b\n", a_out, valid_out);
        
        #20;
        
        $display("========================================");
        $display("  Debug Tests Complete");
        $display("========================================");
        $finish;
    end
endmodule
