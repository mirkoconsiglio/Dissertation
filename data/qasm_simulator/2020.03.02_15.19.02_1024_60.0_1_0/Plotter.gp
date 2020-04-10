set terminal pdfcairo enhanced color dashed font "Times New Roman, 10" rounded size 16 cm, 9.6 cm

command = "mkdir plots"
system command
dir = "plots"

data = "data.csv"
set datafile separator "\t"

theta = pi/3

f(x) = exp(2 * x * log(cos(theta)))
g(x) = 1
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
set yrange[*:1]
set ytics 0.00025

set output dir."/fidelity.pdf"
plot data using 1:3 with points title "Calculated", g(x) with lines title "Theoretical"

set ylabel "Tangle"
set xlabel "Fidelity"
set yrange[0:1]
set xrange[*:1]
set ytics 0.1
set xtics 0.00025
set key top left inside

set output dir."/tangle-fidelity.pdf"
set arrow from 1,0 to 1,1 nohead lc rgb "#2166AC" 
plot data using 3:2 with points title "Calculated",  1/0 title "Theoretical" rgb "#2166AC"
