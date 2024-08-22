version=$1

dir=$PWD

if [ "$version" = "root" ]; then
    cd src
    make generate_root
    ./run_root
    make clean

elif [ "$version" = "hepmc" ]; then
    cd src
    make generate_hepmc
    ./run_hepmc
    make clean

else
    echo "Please use ./run.sh {root|hepmc}"
fi

cd $dir
