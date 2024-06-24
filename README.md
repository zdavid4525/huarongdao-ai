# huarongdao-ai

Run with:

```
python3 hrd.py --algo <algo> --inputfile <input file> --outputfile <output file>    
```

 - <algo> specifies the type of search algorithm: astar or dfs
 - <input file> specifies a plain-text input file to read the puzzle's initial state from
 - <output file> specifies a plain-text output file containing the solution found by the search algorithm

Sample usages are:

```
python3 hrd.py --algo astar --inputfile hrd5.txt --outputfile hrd5sol_astar.txt
or
python3 hrd.py --algo dfs --inputfile hrd7.txt --outputfile hrd7sol_dfs.txt
```
