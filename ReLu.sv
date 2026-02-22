`timescale 1ns / 1ps

// ReLU Activation with Bias Addition
module ReLu #(parameter BITS=32, COUNTER_END=3, B_BITS=32)
(
  input clk,
  input reg signed [BITS+24:0] mult_sum_in,
  input reg [31:0] counter,
  input activation_function,
  output reg signed [BITS + 8:0] neuron_out
);

  wire signed [BITS+24:0] with_bias;
  wire signed [BITS + 8:0] after_relu;

  // Add bias
  assign with_bias = mult_sum_in;
  
  // Apply ReLU: max(0, x)
  assign after_relu = (with_bias < 0) ? 0 : with_bias[BITS + 8:0];

  always @(posedge clk) begin
    neuron_out <= after_relu;
  end

endmodule
