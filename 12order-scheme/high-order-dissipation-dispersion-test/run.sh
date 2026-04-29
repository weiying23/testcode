
rm test


clang -o test spectral_analysis.c -lm


chmod 777 test


./test

sleep 3s

python plot_spectrum.py


