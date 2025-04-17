this is a simple python script for converting a Nondeterministic Finite Automaton (AFN) into a Deterministic Finite Automaton (AFD).

## example usage

```sh
  python .\automate_conversion.py -i .\afn.table -o ./afd.table --initial 1 --final 6   
```

## help ouput
```sh
usage: automate_conversion.py [-h] [-i INPUT] [-o OUTPUT] [-d] [--debug-output DEBUG_OUTPUT] [--initial INITIAL]
                              [--final FINAL [FINAL ...]] [--example]

Convert AFN to AFD using Thompson algorithm

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file containing AFN table
  -o OUTPUT, --output OUTPUT
                        Output markdown file for AFD result
  -d, --debug           Enable debug output
  --debug-output DEBUG_OUTPUT
                        File to write debug information
  --initial INITIAL     Initial state for the AFN
  --final FINAL [FINAL ...]
                        Final states for the AFN
  --example             Use the built-in example AFN
```
