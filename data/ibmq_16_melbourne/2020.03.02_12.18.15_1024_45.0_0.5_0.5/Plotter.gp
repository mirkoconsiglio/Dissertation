set terminal pdfcairo enhanced color dashed font "Times New Roman, 10" rounded size 16 cm, 9.6 cm

command = "mkdir plots"
system command
dir = "plots"

data = "data.csv"
set datafile separator "\t"

theta = pi/4

f(x) = exp(2 * x * log(cos(theta)))
g(x) = (1 + exp(x * log(cos(theta)))) / 2
h(x) = (2 * x - 1) ** 2

set ylabel "Tangle"
set xlabel "Collision Number"
set ytics 0.1
set xtics 1
set grid
set key top right inside

set output dir."/tangle.pdf"
plot data using 1:2 with points title "Numerical", f(x) with lines title "Theoretical"

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