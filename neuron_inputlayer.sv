`timescale 1ns / 1ps


module neuron_inputlayer #(
    parameter NEURON_WIDTH = 3,
    parameter NEURON_BITS  = 15,
    parameter COUNTER_END  = 3,
    parameter B_BITS       = 15
)(
  input clk,
  input rstn,
  input activation_function,
  input signed [31:0] weights [0:NEURON_WIDTH],
  input signed [NEURON_BITS:0] data_in [0:NEURON_WIDTH],
  input signed [B_BITS:0] b,
  input [31:0] counter,
  output signed [NEURON_BITS + 8:0] data_out
  );
  
  wire signed [31:0] bus_w;
  wire signed [NEURON_BITS:0] bus_data;
  wire signed [NEURON_BITS+16:0] bus_mult_result;
  wire signed [NEURON_BITS+24:0] bus_adder;
  wire enable_second_layer;
  
  register #( .WIDTH(NEURON_WIDTH), .BITS(31)) RG_W(
    .clk (clk),
    .rstn (rstn),
    .data (weights),
    .counter (counter),
    .value (bus_w)
  );
  
  register #( .WIDTH(NEURON_WIDTH), .BITS(NEURON_BITS)) RG_X(
    .clk (clk),
    .rstn (rstn),
    .data (data_in),
    .counter (counter),
    .value (bus_data)
  );
  
  multiplier #(.BITS(NEURON_BITS)) MP1
  (
    .clk (clk),
    .rstn (rstn),
    .counter (counter),
    .w (bus_w),
    .x (bus_data),
    .mult_result (bus_mult_result)
  );
  
  adder #(.BITS(NEURON_BITS)) AD1(
    .clk (clk),
    .rstn (rstn),
    .counter (counter),
    .value_in (bus_mult_result),
    .value_out (bus_adder));
  
  ReLu #(.BITS(NEURON_BITS), .COUNTER_END(COUNTER_END), .B_BITS(B_BITS)) activation_and_add_b(
    .clk (clk),
    .mult_sum_in (bus_adder),
    .counter (counter),
    .activation_function(activation_function),
    .b (b),
    .neuron_out (data_out)
  );
    
    
  
endmodule