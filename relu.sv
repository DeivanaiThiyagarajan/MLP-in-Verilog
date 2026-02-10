// ReLU Activation Function
// Applies f(a) = max(0, a) to each input element
// Combinational logic - no state machine needed

module relu #(
    parameter INPUT_WIDTH = 3,
    parameter DATA_WIDTH = 16
) (
    input logic signed [DATA_WIDTH-1:0] a_in [INPUT_WIDTH-1:0],
    output logic signed [DATA_WIDTH-1:0] a_out [INPUT_WIDTH-1:0]
);

    genvar i;
    generate
        for (i = 0; i < INPUT_WIDTH; i = i + 1) begin : relu_gen
            assign a_out[i] = (a_in[i] < 0) ? 0 : a_in[i];
        end
    endgenerate

endmodule
