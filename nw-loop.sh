
help=false

for flag in "$@"
do
    case $flag in
        -h|--help)
            echo "\nUsage: compiles and runs the project in a loop  using a predetermined set of values.\n"
            echo "Options:"
            echo "========\n"
            echo "-h, --help        --- Shows a list of available flags and explanation of each command"
            echo "-a, --all         --- Runs code along with all kernels (Does nothing right now)"
            echo "-0                --- Runs kernel0 (Does nothing right now)"
            echo "-1                --- Runs kernel1 (Does nothing right now)"
            echo "-2                --- Runs kernel2 (Does nothing right now)"
            echo "-3                --- Runs kernel3 (Does nothing right now)"
            help=true
        shift
        ;;
        -0)
        shift
        ;;
        -1)
        shift
        ;;
        -2)
        shift
        ;;
        -3)
        shift
        ;;
        *)
            break
        ;;
    esac
done

if [ ! $help ]
then
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

fi
