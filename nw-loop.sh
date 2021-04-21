
help=false

kernel0=""
kernel1=""
kernel2=""
kernel3=""

title="n, cpu (ms)"

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
            kernel0="-0"
            title="$title, kernel 0 (ms)" 
        shift
        ;;
        -1)
            kernel1="-1"
            title="$title, kernel 1 (ms)" 
        shift
        ;;
        -2)
            kernel2="-2"
            title="$title, kernel 2 (ms)" 
        shift
        ;;
        -3)
            kernel3="-3"
            title="$title, kernel 3 (ms)" 
        shift
        ;;
        *)
            break
        ;;
    esac
done

    make

    echo $title  > benchmark.csv

    for i in 10 100 1000 10000; do
        j=$(($i/10))
        while [ $j -lt $i ]; do 
            printf "$j, " >> benchmark.csv
            output=`./nw -N $j $kernel0 $kernel1 $kernel2 $kernel3 | grep -E "CPU|version" | grep -oE "[0-9]+\.[0-9]+"`
            csv=`echo $output | cut -d \  -f 1`
            if [ $kernel0 != "" ] 
            then
                csv="$csv, `echo $output | cut -d \  -f 2`"
            fi
            if [ $kernel1 != "" ]
            then 
                csv="$csv, `echo $output | cut -d \  -f 3`"
            fi
            if [ $kernel2 != "" ]
            then 
                csv="$csv, `echo $output | cut -d \  -f 4`"
            fi
            echo $csv >> benchmark.csv
            j=$(($j+$i/10))
        done
    done

    j=10000
    while [ $j -le 50000 ]; do 
        printf "$j, " >> benchmark.csv
        output=`./nw -N $j $kernel0 $kernel1 $kernel2 $kernel3 | grep -E "CPU|version" | grep -oE "[0-9]+\.[0-9]+"`
        csv=`echo $output | cut -d \  -f 1`
        if [ $kernel0 != "" ]
        then
            csv="$csv, `echo $output | cut -d \  -f 2`"
        fi
        if [ $kernel1 != "" ]
        then 
            csv="$csv, `echo $output | cut -d \  -f 3`"
        fi
        if [ $kernel2 != "" ]
        then 
            csv="$csv, `echo $output | cut -d \  -f 4`"
        fi
        echo $csv >> benchmark.csv
        j=$(($j+5000))
    done

    make clean

