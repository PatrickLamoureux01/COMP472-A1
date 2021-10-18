Link to repo A1: https://github.com/PatrickLamoureux01/COMP472-A1

The project is divided into 2 .py files. They were coded with PyCharm and are made to be run on PyCharm

Part 1:
1. Make sure you have all external libraries listed at the top of the program. The program requires them to run.

2. Make sure that the BBC data folder location is updated in the APP_FOLDER variable on line 16.

3. Run the program and the output will be written to bbc-performance.txt


Part 2:
1.Make sure you have all external libraries listed at the top of the program. The program requires them to run.

2. Find these variables
	my_csv = pandas.read_csv("C:\\Users\\Pub\\Desktop\\AI Mini Project 1\\drug200.csv")
    pdf = PdfPages('C:\\Users\\Pub\\Desktop\\AI Mini Project 1\\drug-distribution.pdf')
	
They should be around lines 21 and 22.
Change the URL here to wherever you're keeping your drug200.csv and where you want your drug-distribution.pdf created.

3. All 7 methods are found near the bottom. The first plotting() is for creating the graphs printed in
the pdf at URL in step 2. The other 6 are related to the project specificiations and each are
a way to test the data. All 7 run by default, comment out any of them if you wish to focus on certain methods.

4.By default, the 6 methods (other than plotting) print out a classification report. There are also print statements in
each that are commented out. These include information such as "score" and the "confusion matrix". If you wish to also
see this data in the output, uncomment these print statements.

5. In TopDT() and TopMLP() at the very end, there is a commented out block of code. This is if you wish to compare all
the different result from Gridsearch. The first two lines output to the console, though since the data is so large, you may
want to view it in a file. If so, uncomment the rest of the code and make sure f = open("C:\\Users\\Pub\\Desktop\\AI Mini Project 1\\someData.txt", "a")
is changed to whatever desired URL you want. Also the file must be created beforehand.

6.Note* TopMLP() takes a few seconds to run. It is not an error if you do not see the output immediately.
