`timescale 1ns / 1ps

// Sequential Multiplier for MAC operation
module multiplier #(parameter BITS=32)
(
  input clk,
  input rstn,
  input [31:0] counter,
  input signed [31:0] w,
  input signed [BITS:0] x,
  output reg signed [BITS+16:0] mult_result
);

  always @(posedge clk) begin
    if (!rstn) begin
      mult_result <= 0;
    end
    else begin
      $display("w:%d and x:%d", w, x);
      mult_result <= w * x;
      $display("multiply_result:%d", mult_result);
    end
  end

endmodule
