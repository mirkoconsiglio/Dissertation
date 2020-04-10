set terminal pdfcairo noenhanced color dashed font "Times New Roman, 10" rounded size 16 cm, 9.6 cm

data = "plot_data.csv"
set datafile separator "\t"

stats data every ::::0 using 8 nooutput
theta = STATS_max
gamma = log(cos(theta))

stats data every :::0 using 9 nooutput
a_real = STATS_max
stats data every :::0 using 10 nooutput
a_imag = STATS_max
stats data every :::0 using 11 nooutput
b_real = STATS_max
stats data every :::0 using 12 nooutput
b_imag = STATS_max
sv1 = a_real ** 2 + a_imag ** 2
sv2 = b_real ** 2 + b_imag ** 2

tau(x) = exp(2 * x * gamma)
F_GHZ(x) = (1 + exp(gamma * x)) / 2
F_T(x) = 1 - 2 * sv1 * sv2 * (1 - exp(gamma * x))
f(x) = (1 + sqrt(x)) / 2
g(x) = (1 - sqrt(x)) / 2
floor2(x, n) = floor(x * 10 ** n) * 10 ** (-n)

set title "GHZ State Tangle"
set ylabel "3-Tangle"
set xlabel "Collision Number"
set ytics 0.1
set xtics 1
set grid
set key top right inside

set output "tangle.pdf"
plot data using 1:2 with points title "ibmq\_melbourne\_16", data using 1:5 with points title "qasm\_simulator", tau(x) with lines title "Theoretical"

set title "GHZ State Fidelity"
set ylabel "Fidelity"
set xlabel "Collision Number"
set ytics 0.05

set output "GHZ fidelity.pdf"
plot data using 1:3 with points title "ibmq\_melbourne\_16", data using 1:6 with points title "qasm\_simulator", F_GHZ(x) with lines title "Theoretical"

set title "Teleported State Fidelity"

set output "teleported fidelity.pdf"
plot data using 1:4 with points title "ibmq\_melbourne\_16", data using 1:7 with points title "qasm\_simulator", F_T(x) with lines title "Theoretical"

set title "GHZ Fidelity vs GHZ 3-Tangle"
set ylabel "Fidelity"
set xlabel "Tangle"
set ytics 0.1
set xtics 0.05
set key top left inside

set output "fidelity-tangle.pdf"
plot data using 2:3 with points title "ibmq\_melbourne\_16", data using 5:6 with points title "qasm\_simulator", f(x) with lines lc rgb "#1A9850" title "Theoretical", g(x) with lines notitle lc rgb "#1A9850"