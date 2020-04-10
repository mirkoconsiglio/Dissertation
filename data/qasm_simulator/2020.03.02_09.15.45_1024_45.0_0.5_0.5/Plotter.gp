set terminal pdfcairo enhanced color dashed font "Times New Roman, 10" rounded size 16 cm, 9.6 cm

command = "mkdir plots"
system command
dir = "plots"

data = "data.csv"
set datafile separator "\t"

stats data every ::::0 using 4 nooutput
theta = STATS_min
gamma = log(cos(theta))

stats data every :::0 using 5 nooutput
sv1 = STATS_min
sv2 = STATS_max

f(x) = exp(2 * x * gamma)
g(x) = 1 - 2 * sv1 ** 2 * (1 - exp(gamma * x)) + 2 * sv1 ** 4 * (1 - exp(gamma * x)) 
h(x) = (2 * x - 1) ** 2

set ylabel "Tangle"
set xlabel "Collision Number"
set ytics 0.1
set xtics 1
set grid
set key top right inside

set output dir."/tangle.pdf"
plot data using 1:2 with points title "Numerical", f(x) with lines title sprintf('total %d', theta)

set ylabel "Fidelity"
set xlabel "Collision Number"
set ytics 0.05

set output dir."/fidelity.pdf"
plot data using 1:3 with points title "Calculated", g(x) with lines title "Theoretical"

set ylabel "Tangle"
set xlabel "Fidelity"
set ytics 0.1
set xtics 0.05
set key top left inside

set output dir."/tangle-fidelity.pdf"
plot data using 3:2 with points title "Calculated", h(x) with lines title "Theoretical"