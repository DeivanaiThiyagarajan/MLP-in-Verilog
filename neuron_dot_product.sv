// Neuron: Sequential MAC (Multiply-Accumulate)
// Integer implementation - Output: sum(a_in[i]*w_in[i]) + bias

module neuron_dot_product #(
    parameter INPUT_WIDTH = 3,           // Number of inputs
    parameter DATA_WIDTH = 16,           // Word width
    parameter ACC_WIDTH = 48             // Accumulator width
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

    typedef enum logic [1:0] {
        IDLE = 2'b00,
        COMPUTE = 2'b01,
        DONE = 2'b10
    } state_t;
    
    state_t state, next_state;
    logic signed [ACC_WIDTH-1:0] accumulator, next_accumulator;
    logic [$clog2(INPUT_WIDTH):0] index, next_index;
    logic signed [DATA_WIDTH-1:0] result_pipeline;
    logic output_valid_pipeline;
    
    logic signed [DATA_WIDTH*2-1:0] current_product;
    assign current_product = (index < INPUT_WIDTH) ? (a_in[index] * w_in[index]) : '0;
    
    always_comb
        next_accumulator = accumulator + current_product;
    
    assign next_index = index + 1;
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (valid_in) next_state = COMPUTE;
            COMPUTE: if (next_index >= INPUT_WIDTH) next_state = DONE;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            accumulator <= '0;
            index <= '0;
            result_pipeline <= '0;
            output_valid_pipeline <= 1'b0;
        end else begin
            state <= next_state;
            case (state)
                IDLE: begin
                    output_valid_pipeline <= 1'b0;
                    if (valid_in) begin
                        accumulator <= a_in[0] * w_in[0];
                        index <= 1;
                    end
                end
                COMPUTE: begin
                    accumulator <= next_accumulator;
                    index <= next_index;
                end
                DONE: begin
                    result_pipeline <= accumulator + bias;
                    output_valid_pipeline <= 1'b1;
                    index <= '0;
                end
                default: output_valid_pipeline <= 1'b0;
            endcase
        end
    end
    
    assign a_out = result_pipeline;
    assign valid_out = output_valid_pipeline;

endmodule
