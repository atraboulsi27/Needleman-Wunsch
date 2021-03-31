
# Overview

This code implements the Needleman-Wunsch algorithm for exact string matching.

# Instructions

To compile:

```
make
```

To run:

```
./nw [flags]

```

Optional flags:

```
  -N <N>    specified the size of the strings to match

  -0        run GPU version 0
  -1        run GPU version 1
  -2        run GPU version 2
  -3        run GPU version 3
            NOTE: It is okay to specify multiple different GPU versions in the
                  same run. By default, only the CPU version is run.
```

To run in a loop (only works for cpu now):

```
> chmod +x nw-loop.sh // only the first time
> ./nw-loop.sh
```

## The Global Alignment Problem, and the Needleman-Wunsch Algorithm:

Question 2:

```
Describe the computation that you are planning to parallelize in this project. Your description should tackle the following questions:

What is the problem that the computation solves?
Why is the computation important and where is it used in practice?
How does the computation work? What procedure does it follow to solve the problem?
Use illustrations where needed. Make sure to cite any references that you use.
```

Before talking about the Needleman-Wunsch Algorithm, we must introduce the problem it tries to solve.

In Biology, information is very often represented in the form of strands or sequences of molecules. Here are couple of examples:

#### ***DNA:*** 

The study of DNA is called genetics. It is the most basic storage medium of genetic information in all eukaryotes as well as a lot prokaryotes. DNA is sequenced in a long strand of nucleatides that encodes the instructions required to produce every protein that constitutes a living being. There are four nucleatides, and are denominated by the letters A, T, C & G. They are basically molecules that can bind together to form the double stranded shape of DNA. 

DNA can be thought of as a global, unique identifier that specifies the person you are. How similar two strands of DNA, tells us the relationships between two living beings or species. For example, twins will have an extremely similar DNA sequence. Brothers and sister, less so, humans and mokeys even less so. What is interesting about this, is that this common and global store of information can be to study the evolution of biological systems. By studying the similarities and differences between two of those strands, we establish how closely related two biolgical enitites are. From this we can produce many useful analyses, for example, two viruses that have a very similar DNA sequence might be cured by the very same medicine. 

#### ***Proteins:***:

Proteins could be said to be one of the most important building block of any biological system. Every protein is built from molecules called amino acids. There are 20 of them in total. What is important to know about proteins is that is not their content, but their 3D geometry determines their function. It is therefore, a very interesting problem to be able to determine the geometry of a protein. This shape is called a protein's fold, and is generally described by three interconnected structures. First, there is the primary structure which consists of the sequence of amino acids. Next, there is the secondary structure which is specified by the hydrogen bonds that form between these amino acids, forming shapes like alpha-helixes, and beta-sheets. Then, there is the tertiary structure that that arises from the combination of the previous folds and random interactions between the elements of the primary and secondary structures. 

What is interesting about what was said previously is that the tertiary structure of a protein (its 3D geometry), is entirely determined by its primary structure (the strand of amino acids). Therefore, an important question arises: given a random string of amino acids, are you able to predict the 3D shape that a protein will take? This has been shown to be an NP-Complete problem. As such, we cannot, yet, predict the shape of any protein fast enough from a given strand of amino acids. However, what we know, is that very similar protein strands might fold similarly locally. Therefore, if you know how a particular strand folds, you maybe could use this information to make general predictions or guesses about how another similar strand might generally fold (However, this method usually fails in practice).

#### ***The Global Alignment Problem:***

However, we are computer scientists, we prefer thinking of things in terms of data structures and algorithms. What does all of this sounds like to us? Well, a strand of only four possible characters sounds like a string or an array of base 4 numbers. Similarly, a strand of 20 possible characters, sounds like a string or array of base 20 characters. If the biologists were to give us the comparison criterias, we might be able to determine a very fast way of comparing these strings. We could therefore, analyse a large amount of data, extracting value from it. As it happens to be, this problem is called the alignment problem.

There are two types of alignment methods, local alignment and global alignment. In our case, we are interested in global alignment. This method is very useful when comparing two very similar strands. This problem states, given two strings, align the elements between each of those strings such that moving any element in any string would result in a lower or equal match rate. You are allowed to add spaces between two elements in the two strings to get a better match. In biology, theses spaces might be interesting to analyse mutations that could have arisen, they are alled indel. As we can see this problem is very similar, to a problem we know all too well in computer science, and that is the Longest Common Sub-sequence (LCS).

In a more particular way, we are interested in the Needleman-Wunsch Algorithm. This algorithm solves the issue of global alignment by implementing a scoring system for the match rate, using a dynamic programming approach that is quite similar to the LCS Problem. As we know, dynamic programming requires two preliminary conditions about a problem to be fullfiled in order work. These conditions are optimal substructure, and overlapping sub-problems. So, how are, in our case, these requirements met?

* <ins>Optimal Sub-Stucture</ins>:

In our problem, optimal sub-structure presents itself as follows: If I know the optimal solution of a sequence of two strings, then I know that by adding one new character to each string, there are three possible solutions to the optimal alignment of the newly formed strings. The first, is inserting a gap in the first string and adding the new character for the second string, the second option is adding the new character for the first string and a gap in the second string, and the third is inserting the new character for both strings, as is, may it be a match or a mismatch. To decide which option to choose, we rely on a basic scoring system. What this does, is assign a grade to each possible eventuality based on a pre-determined schema. In practice it is common to assign a grade of -1 to any gap or mismatch, and a grade of 1 to a match. When the grades are assigned, the best match is the one that has the highest grade.

* <ins>Overlapping Sub-Problems and Memoization:</ins>:

In our problem, we can show that given two strings of characters, it is possible to determine the best global alignment by deconstructing the problem into simpler sub-problems. As was shown previously, we can determine an optimal solution for the addition of one character to an already optimal sequence alignment. Therefore, by scanning sequences one by one, we can divide the problem into simpler sub-problems that are essantially the same as solving the optimal sub-structure problem. In our case, the optimal alignment can be determined by a memoization step that saves the grade of the current optimal solution into an array, by calculating it from the grade assigned to each possible solution to a previous sub-problem. Since the problems are decreasing in size until the two sub-strings reach a length of 0, the algorithm terminates correctly, filling an array of scores for each sub-sequence. Finally, a traceback step is required to extract the best global alignment from the array.  

What makes the Needleman-Wunsch Algorithm different from the LCS dynamic algorithm, is the traceback step. In fact, to fill the matrix we use the exact same steps.

However, in our case we are not interested in the traceback algorithm as we will concentrate on determining the grade of each sub-problem, and the filling of the memoization matrix.

## Code Description:

### The Needleman-Wunsch Algorithm on the CPU:

The Needleman-Wunsch algorithm requires us to fill an array with sequences of scores in order to determine the correct global alignment between two strings of characters.

```C++
 1    void nw_cpu(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N) {
 2        for(int q = 0; q < N; ++q) {
 3            for (int r = 0; r < N; ++r) {
 4                // Get neighbors
 5                int top     = (q == 0)?((r + 1)*DELETION):(matrix[(q - 1)*N + r]);
 6                int left    = (r == 0)?((q + 1)*INSERTION):(matrix[q*N + (r - 1)]);
 7                int topleft = (q == 0)?(r*DELETION):((r == 0)?(q*INSERTION):(matrix[(q - 1)*N + (r - 1)]));
 8                // Find scores based on neighbors
 9                int insertion = top + INSERTION;
10                int deletion  = left + DELETION;
11                int match     = topleft + ((query[q] == reference[r])?MATCH:MISMATCH);
12                // Select best score
13                int max = (insertion > deletion)?insertion:deletion;
14                max = (match > max)?match:max;
15                matrix[q*N + r] = max;
16            }
17        }
18    }
```

It starts by looping over the two strands on line 2 and 3. 

#### ***Extracting the values of the neighbors:***

During each iteration, the algorithm first needs to get the values of its neighbors in the matrix, specifically the value above (line 5), to the left (line 6), and to the upper left diagonal (line 7), relative to the current value. For each value that needs to be extracted here is the procedure:

* <ins>top:</ins> If the current index in the query string is at position 0, then the value that needs to be extracted is the previous reference index plus 1 times -1. Otherwise, take the value of the matrix from the position above.

* <ins>left:</ins> If the current index in the reference string is at position 0, then the value that needs to be extracted is the previous query index plus 1 times -1. Otherwise, take the value of the matrix from the position left to current value.

* <ins>topleft:</ins> If the current index in the query string is at position 0, then the value that needs to be extracted is the previous reference index plus 1 times -1. Otherwise, if the current index in the reference string is at position 0, then the value that needs to be extracted is the previous query index plus 1 times -1. Otherwise, take the value of the matrix from the position upper-left diagonal to current value.

#### ***Assigning a score to each possible alignment case:***

After that, we need to assign a score to each possible entry in the current position of the matrix. There are four such possibilities: An insertion (line 9), a deletion (line 10), a match (line 11), and a mismatch (line 11).

* <ins>Insertion</ins>: An insertion represents the fact that a gap needs to be inserted to get a better global alignment. Its score is computed by adding the top value to the score of an insertion (-1 in our case).

* <ins>Deletion:</ins> A Deletion also represents the fact that a gap needs to be inserted in order to get a better global alignment. Its score is computed by adding the top value to the score of a deletion (-1 in our case).

* <ins>Match & Mismatch:</ins> A match or a mismatch represents the fact that current position in reference string may contain or not the same value as the query string. The score in this case is computed in the following way: Add the topleft value to the evaluation of the following expression ( If the value contained at the current position in the query string is equal to the value contained at the current position in the reference string, then return a score of a match (+1) else return the score of a mismatch (-1) )

#### ***Comparing the resulting scores:***

Finally, we need to evaluate the best possibility for the alignment at this position (lines 13 & 14). We do that by comparing the scores we computed previously for the insertion, deletion, match, and mismatch.

* <ins>Max of insert and delete:</ins> First we compare the insertion score to the deletion score (line 13), and take the maximum between them.

* <ins>Max of match and previous max:</ins> Then we compare the result of the match scoring (can also represent a mismatch in our case) to the result of the max operation done previously (line 14), and take the maximum between them.

In the end we save the result of the previous operation into the output array (line 15).

## Complexity Analysis:

### Needleman-Wunsch Algorithm on the CPU:

Question 5 here.

<img src="res/nw-cpu-plot.png" alt="A run of the Needleman-Wunsch algorithm on the cpu" width="600">
