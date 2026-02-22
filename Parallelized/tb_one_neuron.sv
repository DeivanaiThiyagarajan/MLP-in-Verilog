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
  logic valid_out;

  longint expected;

  // -------------------------------
  // Clock
  // -------------------------------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // -------------------------------
  // DUT
  // -------------------------------
  neuron_dot_product_parallel #(
      .INPUT_WIDTH(INPUT_WIDTH),
      .DATA_WIDTH(DATA_WIDTH),
      .ACC_WIDTH(ACC_WIDTH)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .a_in(a_in),
      .w_in(w_in),
      .bias(bias),
      .valid_in(valid_in),
      .valid_out(valid_out),
      .a_out(a_out)
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

      wait(valid_out);
    end
  endtask

  // -------------------------------
  // Directed Tests
  // -------------------------------
  initial begin
    @(posedge rst_n);

    // ---- Test 1 simple
    a_in='{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; w_in='{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; bias=0;
    run_test("simple");

    $display("ALL TESTS DONE");
    $finish;
  end

  // -------------------------------
  // VCD dump
  // -------------------------------
  initial begin
    $dumpfile("neuron_parallel.vcd");
    $dumpvars(0,tb_neuron_parallel);
  end

endmodule
