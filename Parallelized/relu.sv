module relu_pipe #(
    parameter DATA_WIDTH = 16
)(
    input logic clk,
    input logic rst_n,
    input logic valid_in,

    input  logic signed [DATA_WIDTH-1:0] in,

    output logic signed [DATA_WIDTH-1:0] out,
    output logic valid_out
);

logic signed [DATA_WIDTH-1:0] relu_pipeline;
logic output_valid_pipeline;

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        relu_pipeline <= '0;
        output_valid_pipeline <= 0;
    end else begin
        if (valid_in) begin
            relu_pipeline <= in[DATA_WIDTH-1] ? '0 : in; // truncate if needed
            output_valid_pipeline <= 1'b1;
        end else begin
            output_valid_pipeline <= 1'b0;
        end
    end
end

assign out = relu_pipeline;
assign valid_out = output_valid_pipeline;

endmodule