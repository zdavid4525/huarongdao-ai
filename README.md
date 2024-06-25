# huarongdao-ai

Each state is a grid of 20 characters which has 5 rows with 4 characters per row.

 - The empty squares are denoted by "."
 - The 2x2 piece is denoted by "1"
 - The single pieces are denoted by "2"
 - A horizontal 1x2 piece is denoted by "<" on the left and ">" on the right
 - A vertical 1x2 piece is denoted by "^" on the top and "v" on the bottom


Run with:

```
python3 hrd.py --algo <algo> --inputfile <input file> --outputfile <output file>    
```

 - algo specifies the type of search algorithm: astar or dfs
 - input file specifies a plain-text input file to read the puzzle's initial state from
 - output file specifies a plain-text output file containing the solution found by the search algorithm

Sample usages are:

```
python3 hrd.py --algo dfs --inputfile hrd5.txt --outputfile hrd5sol_dfs.txt
or
python3 hrd.py --algo astar --inputfile hrd7.txt --outputfile hrd7sol_astar.txt
```
