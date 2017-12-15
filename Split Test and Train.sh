if [ ! -d $2 ];
then
mkdir $2
fi
cd $1
for folder in *
do
	x=$2"/"$folder
	if [ ! -d $x ]
	then
		mkdir $x
	fi
	cd $1
	cd $folder
	count=0
	for video in *
	do
		if [ $count -eq 10 ]
		then
			break
		fi
		mv $video $2"/"$folder
		count=$(($count+1))
	done	
done