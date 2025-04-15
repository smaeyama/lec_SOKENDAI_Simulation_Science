rm -rf */.ipynb_checkpoints/
find ./ -type f -name *.nc  | xargs rm -f
find ./ -type f -name *.mp4  | xargs rm -f
find lec06_mpi/ -type f -name *.py.?  | xargs rm -f
find lec07_data_science/ -type f -name *.py.?  | xargs rm -f
#jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace */*.ipynb
jupyter nbconvert --to python */*.ipynb

