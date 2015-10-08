Hi, Professor.

Here are some notes for our project software installation.

Even though we provided the instructions of installation in our report, there might be some problems if the llvm is compiled under Mac OS X or Windows. You can try follow the instruction in the appendix A under Ubuntu 12.04 or 14.04 LTS with GCC 4.6, clang 3.0 or GCC 4.7 (according to my own experiences)
If you want more test cases you can go download the LLVM 3.0 test-suites http://llvm.org/releases/3.0/test-suite-3.0.tar.gz, and copy it into the "<root>/project" folder.
Files in the folder "Makefiles" are the Makefiles we wrote in case you need to run the whole test cases and it will run and collects the statistics and generate reports. You need to copy the files to "<root>/project/test-suite-3.0/". TEST.idem.* are for their original implementation, TEST.idemGA.* are for our Genetic Algorithms, TEST.idemMC.* are for our Monte Carlo implementation.
For example, if you want to run all the "SingleSource" test cases of our genetic implementation, you can run go to the folder "<root>/project/test-suite-3.0/SingleSource",and type "make TEST=idemGA report". It will generate the results on the screen and generate the report.txt file.

Thanks for reading.
If you have any questions regarding installation and have the program running, please email Raphael or me.

Best regards,
Shuyi