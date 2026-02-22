`timescale 1ns/1ps

module tb_neuron_parallel;

  // -------------------------------
  // Parameters
  // -------------------------------
  parameter INPUT_WIDTH = 10;       // increased from 3 → 8
  parameter DATA_WIDTH  = 16;
  parameter ACC_WIDTH   = 48;
  parameter CLK_PERIOD  = 10;

  // -------------------------------
  // Signals
  // -------------------------------
  logic clk;
  logic rst_n;
  logic valid_in;

  logic signed [DATA_WIDTH-1:0] a_in [INPUT_WIDTH-1:0];
  logic signed [DATA_WIDTH-1:0] w_in [INPUT_WIDTH-1:0];
  logic signed [DATA_WIDTH-1:0] bias;

  logic signed [DATA_WIDTH-1:0] a_out;
  logic signed [DATA_WIDTH-1:0] relu_out;
  logic valid_out_neuron;
  logic valid_out_relu;

  longint expected;

  // -------------------------------
  // Clock
  // -------------------------------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // -------------------------------
  // DUT
  // -------------------------------
  top_neuron_relu #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .a_in(a_in),
        .w_in(w_in),
        .bias(bias),
        .relu_out(relu_out),
        .valid_out(valid_out_relu)
    );

  // -------------------------------
  // Reset
  // -------------------------------
  initial begin
    rst_n = 0;
    valid_in = 0;
    bias = 0;
    #25;
    rst_n = 1;
  end

  // -------------------------------
  // Expected calculator
  // -------------------------------
  //task compute_expected;
    //expected = bias;
    //for(int i=0;i<INPUT_WIDTH;i++)
      //expected += a_in[i]*w_in[i];
    //if(expected<0) begin
        //expected = 0;
    //end
  //endtask

  // -------------------------------
  // Run one test
  // -------------------------------
  task run_test(string name);
    begin
      //compute_expected();

      @(posedge clk);
      valid_in = 1;

      @(posedge clk);
      valid_in = 0;

      wait(valid_out_relu);

      //if(relu_out === expected[DATA_WIDTH-1:0])
        //$display("PASS  %-20s  OUT=%0d", name, relu_out);
      //else
        //$display("FAIL  %-20s  OUT=%0d, NEURON_OUT=%d  EXP=%0d", name, relu_out, a_out, expected);
    end
  endtask


  // -------------------------------
  // Directed Tests
  // -------------------------------
  initial begin
    @(posedge rst_n);

    // ---- Test 1 simple
    //a_in='{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; w_in='{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; bias=0;
    //a_in='{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; w_in='{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; bias=10;
    //a_in='{0, 12, 99, 0, -5, 0, -15, 20, 83, 0}; w_in='{12, 450, 0, -10, 78, 0, 66, 101, 0, 7}; bias=10;
    //a_in='{10, 2, 99, -9, 5, 50, -105, 20, 83, 39}; w_in='{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; bias=0;
    a_in='{10, 2, 99, -9, 5, 50, -105, 20, 83, 39}; w_in='{-10, 12, 89, 300, 2, 9, 56, 12, 7, 107}; bias=0;
    
    
    run_test("simple");

    $display("ALL TESTS DONE");
    $finish;
  end

  // -------------------------------
  // VCD dump
  // -------------------------------
  initial begin
    $dumpfile("neuron_parallel_relu.vcd");
    $dumpvars(0,tb_neuron_parallel);
  end

endmodule
