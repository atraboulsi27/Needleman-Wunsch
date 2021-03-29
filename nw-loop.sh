
make

echo "# n, time (ms)"  > cpu.csv

for i in 10 100 1000 10000 50000; do
    j=$(($i/10))
    while [ $j -lt $i ]; do 
        printf "$j, " >> cpu.csv
        ./nw -N $j | grep -oE "[0-9]+\.[0-9]+" >> cpu.csv
        j=$(($j+$i/10))
    done
done

printf "50000, " >> cpu.csv
./nw -N 50000 | grep -oE "[0-9]+\.[0-9]+" >> cpu.csv

make clean
