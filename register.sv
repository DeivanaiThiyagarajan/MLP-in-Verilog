`timescale 1ns / 1ps

module register #(
    parameter WIDTH = 3,
    parameter BITS  = 15
)
(
  input clk,
  input rstn,
  input signed [BITS:0] data [0:WIDTH],
  input [31:0] counter,
  output reg signed [BITS:0] value
  );
  
  always @(posedge clk) begin
    if (!rstn) begin
      value <= 0;
    end
    else begin
    $display("counter_reg:%d and value_reg:%d", counter, value);
    value <= data[counter];
  end
  end
endmodule