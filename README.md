
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

## The Global Alignment Problem, and the Needleman-Wunsch Algorithm:

Question 2:

1-The needleman wunsch algorithm solves the issue of the global alignment, which is the alignment of two entire sequences with eachother.

2-This computation is used in the field of bioinformatics, in which similarities between two  amino acid sequences must be located to study their functional, structural and evolutionary relationships. The needleman-wunsch algorithm does so in an optimal maner, highlighting its importance for large sequences.


## Code Description:

### Needleman-Wunsch Algorithm on the CPU:

```C++ {.line-numbers}
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

The Needleman-Wunsch algorithm requires us to fill an array with sequences of scores in order to determine the correct global alignment between two strings of characters. Therefore, it starts by looping over the two strands on line 2 and 3. 

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
