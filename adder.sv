`timescale 1ns / 1ps

// Sequential Adder for accumulating MAC results
module adder #(parameter BITS=32)
(
  input clk,
  input rstn,
  input [31:0] counter,
  input signed [BITS+16:0] value_in,
  output reg signed [BITS+24:0] value_out
);

  reg signed [BITS+24:0] accumulator;

  always @(posedge clk) begin
    if (!rstn) begin
      accumulator <= 0;
      value_out <= 0;
    end
    else begin
      // Accumulate and immediately output the new sum
      $display("accumulator: %d value_in:%d value_out:%d", accumulator, value_in, value_out);
      accumulator <= accumulator + value_in;
      value_out <= accumulator;
    end
  end

endmodule
