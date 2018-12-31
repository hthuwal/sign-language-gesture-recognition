if [ ! -d $2 ];
then
	mkdir $2
fi

source_dir="$(realpath $1)"
dest_dir="$(realpath $2)"

rm -rf dest_dir

for folder in "$source_dir"/*
do
	label="$(basename $folder)"
	echo "Label: "$label
	target="$dest_dir/$label"
	if ! [ -d "$target" ]; then
		mkdir -p "$target"
	fi

	count=0
	for file in "$folder"/*
	do
		file_name="$(basename $file)"
		if [ $count -eq 2 ];then
			break
		fi
		cp "$file" "$target/$file_name"
		count=$((count + 1))
	done
done
