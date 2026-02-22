`timescale 1ns/1ps
module top_neuron_relu #(
    parameter INPUT_WIDTH=10, 
    parameter DATA_WIDTH=16, 
    parameter ACC_WIDTH=48
) (
    input logic clk, rst_n, valid_in,
    input logic signed [DATA_WIDTH-1:0] a_in [INPUT_WIDTH-1:0],
    input logic signed [DATA_WIDTH-1:0] w_in [INPUT_WIDTH-1:0],
    input logic signed [DATA_WIDTH-1:0] bias,
    output logic signed [DATA_WIDTH-1:0] relu_out,
    output logic valid_out
);
    logic signed [DATA_WIDTH-1:0] neuron_out;
    logic neuron_valid;

    neuron_dot_product_parallel #(.INPUT_WIDTH(INPUT_WIDTH), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) neuron_inst (
        .clk(clk), .rst_n(rst_n), .a_in(a_in), .w_in(w_in), .bias(bias),
        .valid_in(valid_in), .valid_out(neuron_valid), .a_out(neuron_out)
    );

    relu_pipe #(.DATA_WIDTH(DATA_WIDTH)) relu_inst (
        .clk(clk), .rst_n(rst_n), .valid_in(neuron_valid), .in(neuron_out),
        .out(relu_out), .valid_out(valid_out)
    );
endmodule