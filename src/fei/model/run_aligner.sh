#!/bin/bash

input_dir=/home/user/Data/Proxy/gold
scripts_dir=/home/user/Tools/jamr/scripts/

# configure
`. $scripts_dir/config_new.sh`

# remove aligned files
for input_file in `find $input_dir -type f -name aligned-\*`; do
    rm -f $input_file
done

# align all proxy files
for input_file in `find $input_dir -type f -name \*-proxy\*`; do
    filename=`basename $input_file`
    dirname=`dirname $input_file`
    output_file=$dirname/"aligned-"$filename
    $scripts_dir/ALIGN.sh < $input_file > $output_file
done
