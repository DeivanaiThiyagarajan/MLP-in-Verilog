`timescale 1ns / 1ps
 
module MAC #( parameter IN_BITWIDTH = 16,
			  parameter OUT_BITWIDTH = 32,
			  parameter B_BITS = 15,
			  parameter COUNTER_END = 3 )
			( input signed [IN_BITWIDTH-1 : 0] a_in,
			  input signed [IN_BITWIDTH-1 : 0] w_in,
			  input signed [B_BITS:0] bias,
			  input [31:0] counter,
			  input en, clk, rstn,
			  output reg signed [OUT_BITWIDTH-1 : 0] out
			);
	reg signed [OUT_BITWIDTH-1:0] mult_out;
	reg signed [OUT_BITWIDTH-1:0] accumulator;
	
	always@(posedge clk) begin
		if (!rstn) begin
			accumulator <= bias;
			out <= 0;
		end
		else if(en) begin
			mult_out = a_in * w_in;
			accumulator <= accumulator + mult_out;  // Accumulate internally
			out <= accumulator;  // Output previous accumulator value
            $display("a_in:%0d, w_in:%0d, mult_out:%0d, accumulator:%0d, counter:%0d and out:%0d", a_in, w_in, mult_out, accumulator, counter, out);
		end
	end
endmodule