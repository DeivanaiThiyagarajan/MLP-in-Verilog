`timescale 1ns / 1ps

module tb_half_adder;

    logic a;
    logic b;
    logic sum;
    logic carry;

    // Instantiate DUT
    half_adder dut (
        .a(a),
        .b(b),
        .sum(sum),
        .carry(carry)
    );

    // Clock-like toggling for inputs
    initial begin
        a = 0;
        b = 0;

        #10;
        a = 1; b = 0; #10;
        a = 0; b = 1; #10;
        a = 1; b = 1; #10;
        a = 0; b = 0; #10;

        $stop;  // stops simulation so UCLI can generate SAIF
    end

    // -----------------------------
    // VCD dump for debugging / PrimeTime
    // -----------------------------
    initial begin
        $dumpfile("half_adder.vcd");
        $dumpvars(0, tb_half_adder);
    end

endmodule
