Instructions to compile and run our program

	Create a folder to host the compiler (to be referenced as <root> ).
	Go to the folder <root>
cd <root>
	Download the GitHub source of LLVM with Idempotence extensions
 git clone https://github.com/mdekruijf/llvm.git 
cd llvm
git checkout idempotence extensions
	Download clang
cd tools
svn co http://llvm.org/svn/llvm-project/cfe/trunk clang -r 149259
	Go to the folder <root>/llvm/lib/codegen
cd <root>/llvm/lib/codegen

	Replace the files MemoryIdempotenceAnalysis.cpp and ConstructIdempotentRegions.cpp with the ones provided in the submission of this report.

	Configure and Make the executable in LLVM¡¯s root folder.

cd <root>/llvm
./configure -disable-optimized
make

The executables will be generated in the folder <root>/llvm/Debug+Asserts/bin

	List the parameters added in the executable llc
cd <root>/llvm/Debug+Asserts/bin
llc -help-hidden

	Compile and analyze C/C++ programs using our program
export PATH=<root>/llvm/Debug+Asserts/bin:$PATH
clang -s -emit-llvm <file>.c -o <file>.bc
llc   -idempotence-construction=size <parameters defined in this report> <file>.bc -o <file>.s


Here are some notes for our project software installation.

Even though we provided the instructions of installation in our report, there might be some problems if the llvm is compiled under Mac OS X or Windows. You can try follow the instruction in the appendix A under Ubuntu 12.04 or 14.04 LTS with GCC 4.6, clang 3.0 or GCC 4.7 (according to my own experiences)
If you want more test cases you can go download the LLVM 3.0 test-suites http://llvm.org/releases/3.0/test-suite-3.0.tar.gz, and copy it into the "<root>/project" folder.
Files in the folder "Makefiles" are the Makefiles we wrote in case you need to run the whole test cases and it will run and collects the statistics and generate reports. You need to copy the files to "<root>/project/test-suite-3.0/". TEST.idem.* are for their original implementation, TEST.idemGA.* are for our Genetic Algorithms, TEST.idemMC.* are for our Monte Carlo implementation.
For example, if you want to run all the "SingleSource" test cases of our genetic implementation, you can run go to the folder "<root>/project/test-suite-3.0/SingleSource",and type "make TEST=idemGA report". It will generate the results on the screen and generate the report.txt file.

Thanks for reading.
If you have any questions regarding installation and have the program running, please email Raphael or me.

Best regards,
Shuyi