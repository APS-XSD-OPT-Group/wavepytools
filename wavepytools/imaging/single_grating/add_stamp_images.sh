#! /bin/bash


# Example: ./add_stamp_images.sh WF*png


FILES=($@)
NFILES=$#


TIMESTEP=5 ## put minutes step here


# echo 'FILES: '$FILES
echo 'NFILES: '$NFILES



for (( i=0; i<=NFILES; i++ ));
do
	((label = $i * $TIMESTEP)) 
	label=$label'_min'
	label=${FILES[i]}
	echo $label': '${FILES[i]}
	convert ${FILES[i]} -gravity Southwest -pointsize 30 -annotate +50+30 $label  temp_${FILES[i]}
done