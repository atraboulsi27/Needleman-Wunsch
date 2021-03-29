
make

echo "n, time (ms)"  > cpu.csv

for i in 10 100 1000 10000; do
    j=$(($i/10))
    while [ $j -lt $i ]; do 
        printf "$j, " >> cpu.csv
        ./nw -N $j | grep -oE "[0-9]+\.[0-9]+" >> cpu.csv
        j=$(($j+$i/10))
    done
done

j=10000
while [ $j -le 50000 ]; do 
    printf "$j, " >> cpu.csv
    ./nw -N $j | grep -oE "[0-9]+\.[0-9]+" >> cpu.csv
    j=$(($j+5000))
done

make clean
