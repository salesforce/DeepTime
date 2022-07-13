for instance in `/bin/ls -d storage/experiments/hp_search/*/*`; do
    echo $instance
    make run command=${instance}/command
done
