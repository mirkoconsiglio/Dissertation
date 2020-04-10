set terminal pdfcairo enhanced color dashed font "Times New Roman, 10" rounded size 16 cm, 9.6 cm

data = "plot_data.csv"
set datafile separator "\t"

stats data every ::::0 using 5 nooutput
theta = STATS_max
gamma = log(cos(theta))

stats data every :::0 using 6 nooutput
sv1 = STATS_min
sv2 = STATS_max

tau(x) = exp(2 * x * gamma)
F_GHZ(x) = (1 + exp(gamma * x)) / 2
F_T(x) = 1 - 2 * sv1 ** 2 * sv2 ** 2 * (1 - exp(gamma * x))
f(x) = (1 + sqrt(x)) / 2
floor2(x, n) = floor(x * 10 ** n) * 10 ** (-n)

set ylabel "Tangle"
set xlabel "Collision Number"
set ytics 0.1
set xtics 1
set grid
set key top right inside

set output "tangle.pdf"
plot data using 1:2 with points title "Numerical", tau(x) with lines title "Theoretical"

set ylabel "Fidelity"
set xlabel "Collision Number"
set ytics 0.05

set output "GHZ fidelity.pdf"
plot data using 1:3 with points title "GHZ Fidelity", F_GHZ(x) with lines title "Theoretical"

y = floor2(4 * sv1 ** 2 * sv2 ** 2, 2) / 10
set ytics y

set output "teleported fidelity.pdf"
plot data using 1:4 with points title "Teleported Fidelity", F_T(x) with lines title "Theoretical"

set ylabel "Fidelity"
set xlabel "Tangle"
set ytics 0.1
set xtics 0.05
set key top left inside

set output "fidelity-tangle.pdf"
plot data using 2:3 with points title "GHZ Fidelity", f(x) with lines title "Theoretical"