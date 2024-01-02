# Install C++/CUDA extensions
for ext in mesh2sdf_cuda sol_nglod; do
    cd $ext && /opt/conda/envs/newCondaEnvironment/bin/python3 setup.py clean --all install --user && cd -
done
