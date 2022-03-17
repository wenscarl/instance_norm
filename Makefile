ll:
	g++ -c -o instance_norm.o instance_norm.cpp
	g++ -shared -o libin.so -fPIC instance_norm.o
	nvcc -o instance_norm.out instance_norm.cu instance_norm.o
	g++ -I eigen-3.4.0 -o instance_norm_eigen.out instance_norm_eigen.cpp instance_norm.o -O2 -mavx2 -DNDEBUG

clean:
	rm -rf *.so *.o *.out
