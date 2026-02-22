`timescale 1ns / 1ps


module neuron_inputlayer_mac #(
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
  wire signed [31:0] bus_mac_out;
  wire mac_enable;
  
  // Enable MAC when counter is active
  assign mac_enable = 1'b1;
  
  // Connect MAC output directly to module output (truncate to match width)
  assign data_out = bus_mac_out[NEURON_BITS + 8:0];
  
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
  
  // MAC module with internal accumulation and bias addition
  MAC #(
    .IN_BITWIDTH(16),
    .OUT_BITWIDTH(32),
    .B_BITS(B_BITS),
    .COUNTER_END(COUNTER_END)
  ) MAC_UNIT (
    .a_in(bus_data),
    .w_in(bus_w[15:0]),
    .bias(b),
    .counter(counter),
    .en(mac_enable),
    .clk(clk),
    .rstn(rstn),
    .out(bus_mac_out)
  );
    
    
  
endmodule
