`timescale 1ns/1ps

module neuron_dot_product_parallel #(
    parameter INPUT_WIDTH = 3,
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH = 48
) (
    input logic clk,
    input logic rst_n,
    
    input logic signed [DATA_WIDTH-1:0] a_in [INPUT_WIDTH-1:0],
    input logic signed [DATA_WIDTH-1:0] w_in [INPUT_WIDTH-1:0],
    input logic signed [DATA_WIDTH-1:0] bias,
    
    input logic valid_in,
    output logic valid_out,
    output logic signed [DATA_WIDTH-1:0] a_out
);

    // Stage 1: Compute products in parallel
    logic signed [DATA_WIDTH*2-1:0] products [INPUT_WIDTH-1:0];
    genvar i;
    generate
        for (i = 0; i < INPUT_WIDTH; i++) begin : MULTS
            assign products[i] = a_in[i] * w_in[i];
        end
    endgenerate

    // Stage 2: Sum products using adder tree
    logic signed [ACC_WIDTH-1:0] sum_products;

    always_comb begin
        sum_products = '0;
        for (int j = 0; j < INPUT_WIDTH; j++) begin
            sum_products = sum_products + products[j];
        end
    end

    // Stage 3: Add bias
    logic signed [ACC_WIDTH-1:0] sum_with_bias;
    always_comb sum_with_bias = sum_products + bias;

    // Stage 4: Pipeline registers (optional)
    logic signed [DATA_WIDTH-1:0] result_pipeline;
    logic output_valid_pipeline;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_pipeline <= '0;
            output_valid_pipeline <= 1'b0;
        end else begin
            if (valid_in) begin
                result_pipeline <= sum_with_bias[DATA_WIDTH-1:0]; // truncate if needed
                output_valid_pipeline <= 1'b1;
            end else begin
                output_valid_pipeline <= 1'b0;
            end
        end
    end

    assign a_out = result_pipeline;
    assign valid_out = output_valid_pipeline;

endmodule
