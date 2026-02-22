`timescale 1ns/1ps

module tb_half_adder_random;

reg a, b;               // Inputs to the half-adder
wire sum, carry;        // Outputs from the half-adder

integer i;

// Instantiate the half-adder
half_adder uut (
    .a(a),
    .b(b),
    .sum(sum),
    .carry(carry)
);

// Generate VCD for waveform and toggle activity
initial begin
    $dumpfile("half_adder_random.vcd");
    $dumpvars(0, tb_half_adder_random);
end

// Random stimulus
initial begin
    a = 0; b = 0;

    // Generate 100 random input combinations
    for (i = 0; i < 100; i = i + 1) begin
        #5  a = $random % 2;  // Random 0 or 1
        #5  b = $random % 2;  // Random 0 or 1
    end

    #10 $stop;  // End simulation
end

// Optional: Monitor signals
initial begin
    $monitor("Time=%0t | a=%b b=%b -> sum=%b carry=%b", $time, a, b, sum, carry);
end

endmodule
