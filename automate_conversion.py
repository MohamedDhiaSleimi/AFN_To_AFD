import sys
import re
import argparse
from collections import defaultdict

class AFN:
    def __init__(self):
        self.states = set()           # Set of states
        self.alphabet = set()         # Input alphabet
        self.transitions = {}         # Dictionary mapping (state, symbol) to a set of states
        self.initial_state = None     # Initial state
        self.final_states = set()     # Set of final states
    
    def add_transition(self, from_state, symbol, to_state):
        """Add a transition from from_state to to_state with symbol."""
        self.states.add(from_state)
        self.states.add(to_state)
        if symbol != 'ε':  # Epsilon is not part of the alphabet
            self.alphabet.add(symbol)
        
        if (from_state, symbol) not in self.transitions:
            self.transitions[(from_state, symbol)] = set()
        self.transitions[(from_state, symbol)].add(to_state)
    
    def set_initial_state(self, state):
        """Set the initial state."""
        self.states.add(state)
        self.initial_state = state
    
    def add_final_state(self, state):
        """Add a final state."""
        self.states.add(state)
        self.final_states.add(state)
    
    def epsilon_closure(self, states):
        """
        Compute the epsilon-closure of a set of states.
        The epsilon-closure of a state s is the set of states reachable from s
        by following epsilon transitions.
        """
        closure = set(states)
        stack = list(states)
        
        while stack:
            state = stack.pop()
            if (state, 'ε') in self.transitions:
                for next_state in self.transitions[(state, 'ε')]:
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
        
        return closure
    
    def move(self, states, symbol):
        """
        Compute the set of states reachable from any state in states
        by following a transition with the given symbol.
        """
        result = set()
        for state in states:
            if (state, symbol) in self.transitions:
                result.update(self.transitions[(state, symbol)])
        return result
    
    def to_markdown(self):
        """Convert the AFN to a markdown representation."""
        md = "# Non-deterministic Finite Automaton (AFN)\n\n"
        
        md += "## States\n"
        md += ", ".join(str(state) for state in sorted(self.states)) + "\n\n"
        
        md += "## Alphabet\n"
        md += ", ".join(sorted(self.alphabet)) + "\n\n"
        
        md += "## Initial State\n"
        md += f"{self.initial_state}\n\n"
        
        md += "## Final States\n"
        md += ", ".join(str(state) for state in sorted(self.final_states)) + "\n\n"
        
        # Create transition table
        md += "## Transitions\n"
        md += "| Etat |" + "".join(f" {symbol} |" for symbol in sorted(self.alphabet) + ['ε']) + "\n"
        md += "|------|" + "---|" * (len(self.alphabet) + 1) + "\n"
        
        # Build transition table for each state
        for state in sorted(self.states):
            row = [f"| {state} |"]
            
            for symbol in sorted(self.alphabet) + ['ε']:
                if (state, symbol) in self.transitions:
                    to_states = sorted(self.transitions[(state, symbol)])
                    row.append(f" {','.join(str(s) for s in to_states)} |")
                else:
                    row.append(" - |")
            
            md += "".join(row) + "\n"
        
        return md


class AFD:
    def __init__(self):
        self.states = set()           # Set of states (each state is a frozenset of AFN states)
        self.alphabet = set()         # Input alphabet
        self.transitions = {}         # Dictionary mapping (state, symbol) to a state
        self.initial_state = None     # Initial state
        self.final_states = set()     # Set of final states
        self.state_aliases = {}       # Dictionary mapping frozenset states to aliases
        
    def add_transition(self, from_state, symbol, to_state):
        """Add a transition from from_state to to_state with symbol."""
        self.states.add(from_state)
        self.states.add(to_state)
        self.alphabet.add(symbol)
        self.transitions[(from_state, symbol)] = to_state
    
    def set_initial_state(self, state):
        """Set the initial state."""
        self.states.add(state)
        self.initial_state = state
    
    def add_final_state(self, state):
        """Add a final state."""
        self.states.add(state)
        self.final_states.add(state)
    
    def assign_state_aliases(self):
        """Assign readable aliases to states."""
        for i, state in enumerate(sorted(self.states, key=lambda x: tuple(sorted(x)))):
            self.state_aliases[state] = f"q{i}"
    
    def get_state_label(self, state):
        """Get the label for a state, either its alias or the state itself."""
        if state in self.state_aliases:
            return f"{self.state_aliases[state]} {sorted(state)}"
        return str(sorted(state))
    
    def print_afd(self, use_aliases=True):
        """Print the AFD in a readable format."""
        if use_aliases and not self.state_aliases:
            self.assign_state_aliases()
            
        print("States:", len(self.states))
        for state in sorted(self.states, key=lambda x: tuple(sorted(x))):
            if use_aliases:
                print(f"  {self.state_aliases[state]}: {sorted(state)}")
            else:
                print(f"  {sorted(state)}")
                
        print("Alphabet:", self.alphabet)
        
        if use_aliases:
            print("Initial state:", self.state_aliases[self.initial_state], sorted(self.initial_state))
            print("Final states:", [self.state_aliases[state] for state in self.final_states])
        else:
            print("Initial state:", sorted(self.initial_state))
            print("Final states:", [sorted(state) for state in self.final_states])
        
        print("Transitions:")
        # Sort transitions for consistent output
        sorted_transitions = sorted(self.transitions.items(), 
                                    key=lambda x: (tuple(sorted(x[0][0])), x[0][1]))
        
        for (state, symbol), next_state in sorted_transitions:
            if use_aliases:
                from_label = self.state_aliases[state]
                to_label = self.state_aliases[next_state]
                print(f"  ({from_label}, {symbol}) -> {to_label}")
            else:
                print(f"  ({sorted(state)}, {symbol}) -> {sorted(next_state)}")

    def print_transition_table(self):
        """Print the AFD as a transition table."""
        if not self.state_aliases:
            self.assign_state_aliases()
            
        # Header
        header = "State | " + " | ".join(sorted(self.alphabet)) + " | Final?"
        print(header)
        print("-" * len(header))
        
        # Sort states for consistent output
        for state in sorted(self.states, key=lambda x: self.state_aliases[x]):
            is_final = "Yes" if state in self.final_states else "No"
            initial_marker = "*" if state == self.initial_state else " "
            row = [f"{initial_marker}{self.state_aliases[state]}"]
            
            for symbol in sorted(self.alphabet):
                if (state, symbol) in self.transitions:
                    next_state = self.transitions[(state, symbol)]
                    row.append(self.state_aliases[next_state])
                else:
                    row.append("-")
                    
            row.append(is_final)
            print(" | ".join(map(str, row)))
    
    def to_markdown(self):
        """Convert the AFD to a markdown representation."""
        if not self.state_aliases:
            self.assign_state_aliases()
            
        md = "# Deterministic Finite Automaton (AFD)\n\n"
        
        md += "## States\n"
        md += "| Alias | States |\n"
        md += "|-------|--------|\n"
        for state in sorted(self.states, key=lambda x: self.state_aliases[x]):
            md += f"| {self.state_aliases[state]} | {sorted(state)} |\n"
        md += "\n"
        
        md += "## Alphabet\n"
        md += ", ".join(sorted(self.alphabet)) + "\n\n"
        
        md += "## Initial State\n"
        md += f"{self.state_aliases[self.initial_state]} {sorted(self.initial_state)}\n\n"
        
        md += "## Final States\n"
        md += ", ".join(f"{self.state_aliases[state]} {sorted(state)}" for state in sorted(self.final_states, key=lambda x: self.state_aliases[x])) + "\n\n"
        
        # Create transition table in the specified format
        md += "## Transitions\n"
        md += "| Etat |" + "".join(f" {symbol} |" for symbol in sorted(self.alphabet)) + "\n"
        md += "|------|" + "---|" * len(self.alphabet) + "\n"
        
        # Build transition table for each state
        for state in sorted(self.states, key=lambda x: self.state_aliases[x]):
            row = [f"| {self.state_aliases[state]} |"]
            
            for symbol in sorted(self.alphabet):
                if (state, symbol) in self.transitions:
                    next_state = self.transitions[(state, symbol)]
                    row.append(f" {self.state_aliases[next_state]} |")
                else:
                    row.append(" - |")
            
            md += "".join(row) + "\n"
        
        return md


def convert_afn_to_afd(afn, debug=False, debug_file=None):
    """
    Convert an AFN to an AFD using the Thompson algorithm (subset construction).
    If debug is True, print steps of the conversion process.
    If debug_file is provided, write debug information to the file.
    """
    afd = AFD()
    
    # Function to write debug info to both console and file if needed
    def debug_print(message):
        if debug:
            print(message)
        if debug_file:
            debug_file.write(message + "\n")
    
    # The initial state of the AFD is the epsilon-closure of the initial state of the AFN
    initial_closure = afn.epsilon_closure({afn.initial_state})
    initial_state = frozenset(initial_closure)
    afd.set_initial_state(initial_state)
    
    if debug or debug_file:
        debug_print("\n=== Conversion Steps ===")
        debug_print(f"Initial state: epsilon-closure({afn.initial_state}) = {sorted(initial_closure)}")
    
    # If the initial state contains a final state of the AFN, it's also a final state in the AFD
    if any(state in afn.final_states for state in initial_state):
        afd.add_final_state(initial_state)
        if debug or debug_file:
            debug_print(f"State {sorted(initial_state)} contains a final state, marking it as final")
    
    # Set of states to process
    unmarked_states = [initial_state]
    marked_states = set()
    
    # Tables for debug output
    if debug or debug_file:
        debug_print("\n=== Subset Construction Steps ===")
        step_num = 1
    
    # Process all states
    while unmarked_states:
        current_state = unmarked_states.pop(0)
        marked_states.add(current_state)
        
        if debug or debug_file:
            debug_print(f"\nStep {step_num}: Processing state {sorted(current_state)}")
            step_num += 1
        
        # For each symbol in the alphabet
        for symbol in sorted(afn.alphabet):
            # Compute the next state by moving from the current state with the symbol
            # and then computing the epsilon-closure
            move_result = afn.move(current_state, symbol)
            next_state = frozenset(afn.epsilon_closure(move_result))
            
            if debug or debug_file:
                debug_print(f"  Symbol {symbol}:")
                debug_print(f"    move({sorted(current_state)}, {symbol}) = {sorted(move_result)}")
                debug_print(f"    epsilon-closure({sorted(move_result)}) = {sorted(next_state)}")
            
            # If the resulting state is not empty
            if next_state:
                # Add the transition to the AFD
                afd.add_transition(current_state, symbol, next_state)
                
                if debug or debug_file:
                    debug_print(f"    Add transition: ({sorted(current_state)}, {symbol}) -> {sorted(next_state)}")
                
                # If the next state is not marked, add it to the list of states to process
                if next_state not in marked_states and next_state not in unmarked_states:
                    unmarked_states.append(next_state)
                    if debug or debug_file:
                        debug_print(f"    New state discovered: {sorted(next_state)}")
                
                # If the next state contains a final state of the AFN, it's also a final state in the AFD
                if any(state in afn.final_states for state in next_state):
                    afd.add_final_state(next_state)
                    if debug or debug_file:
                        debug_print(f"    State {sorted(next_state)} contains a final state, marking it as final")
    
    afd.alphabet = afn.alphabet
    afd.assign_state_aliases()
    
    return afd


def parse_markdown_to_afn(markdown_table, initial_state=None, final_states=None):
    """
    Parse a markdown table representing an AFN (Automaton Finite Non-deterministic)
    
    Args:
        markdown_table (str): Markdown table string
        initial_state: Optional initial state to set
        final_states: Optional list of final states
    
    Returns:
        AFN: An instance of the AFN class with the parsed data
    """
    # Create a new AFN
    afn = AFN()
    
    # Split the table into lines
    lines = markdown_table.strip().split('\n')
    
    # Find the header line (should start with | Etat |)
    header_line = None
    for i, line in enumerate(lines):
        if re.match(r'\s*\|\s*Etat\s*\|', line, re.IGNORECASE):
            header_line = i
            break
    
    if header_line is None:
        raise ValueError("Invalid table format: Header not found or not in the expected format")
    
    # Process the header to get the symbols
    header = lines[header_line]
    header_parts = [part.strip() for part in header.split('|')]
    header_parts = [part for part in header_parts if part]  # Remove empty strings
    
    # The first column is for states, the rest are for symbols
    symbols = header_parts[1:]
    
    # Skip the separator line
    content_start = header_line + 2  # +1 for header, +1 for separator
    
    # Process each line of the table content
    for i in range(content_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            continue
            
        # Extract the state and transitions
        parts = [part.strip() for part in line.split('|')]
        parts = [part for part in parts if part]  # Remove empty strings
        
        if len(parts) < 2:  # Need at least state and one transition
            continue
            
        try:
            state = int(parts[0])
            afn.states.add(state)
            
            # Process transitions for each symbol
            for j, symbol in enumerate(symbols):
                if j + 1 < len(parts):
                    transition_value = parts[j + 1]
                    if transition_value and transition_value != '-':
                        # Split multiple transitions (comma-separated)
                        destinations = [dest.strip() for dest in transition_value.split(',')]
                        for dest in destinations:
                            if dest:
                                try:
                                    dest_state = int(dest)
                                    afn.add_transition(state, symbol, dest_state)
                                except ValueError:
                                    print(f"Warning: Invalid destination state '{dest}' - skipping")
        except ValueError:
            print(f"Warning: Invalid state '{parts[0]}' - skipping row")
    
    # Set initial state if provided, otherwise use the smallest state
    if initial_state is not None:
        afn.set_initial_state(initial_state)
    elif afn.states:
        afn.set_initial_state(min(afn.states))
    
    # Set final states if provided, otherwise use the largest state
    if final_states:
        for state in final_states:
            if state in afn.states:
                afn.add_final_state(state)
    elif afn.states and not afn.final_states:
        afn.add_final_state(max(afn.states))
    
    return afn


def parse_afn_from_file(file_path, initial_state=None, final_states=None):
    """Parse an AFN from a file containing a table."""
    try:
        with open(file_path, 'r',encoding='utf-8') as file:
            content = file.read()
        return parse_markdown_to_afn(content, initial_state, final_states)
    except Exception as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        sys.exit(1)


def example_afn_from_table():
    """Create an example AFN from the provided table format."""
    table = """| Etat | a     | b   | ε     |
|------|-------|-----|-------|
| 1    | 1,2   | 3   | -     |
| 2    | 4     | 3   | 3,4   |
| 3    | 5     | -   | -     |
| 4    | 6     | 5   | 5     |
| 5    | 5,6   | 2   | 6     |
| 6    | -     | 6   | -     |"""
    
    return parse_markdown_to_afn(table, initial_state=1, final_states=[6])


def main():
    parser = argparse.ArgumentParser(description='Convert AFN to AFD using Thompson algorithm')
    parser.add_argument('-i', '--input', help='Input file containing AFN table')
    parser.add_argument('-o', '--output', help='Output markdown file for AFD result')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--debug-output', help='File to write debug information')
    parser.add_argument('--initial', type=int, help='Initial state for the AFN')
    parser.add_argument('--final', type=int, nargs='+', help='Final states for the AFN')
    parser.add_argument('--example', action='store_true', help='Use the built-in example AFN')
    
    args = parser.parse_args()
    
    # Determine input source
    if args.example:
        afn = example_afn_from_table()
        print("Using built-in example AFN:")
    elif args.input:
        print(f"Parsing AFN from file: {args.input}")
        afn = parse_afn_from_file(args.input, args.initial, args.final)
    else:
        # Use the hardcoded example by default
        print("No input specified. Using built-in example AFN.")
        afn = example_afn_from_table()
    
    # Print AFN details
    print("\n=== AFN Definition ===")
    print("States:", afn.states)
    print("Alphabet:", afn.alphabet)
    print("Initial state:", afn.initial_state)
    print("Final states:", afn.final_states)
    print("Transitions:")
    for (state, symbol), next_states in sorted(afn.transitions.items()):
        print(f"  ({state}, {symbol}) -> {next_states}")
    
    # Open debug file if specified
    debug_file = None
    if args.debug_output:
        try:
            debug_file = open(args.debug_output, 'w',encoding='utf-8')
            debug_file.write("# AFN to AFD Conversion Debug Log\n\n")
            debug_file.write("## AFN Definition\n\n")
            debug_file.write(afn.to_markdown())
            debug_file.write("\n")
        except Exception as e:
            print(f"Error opening debug file {args.debug_output}: {e}")
            args.debug_output = None
    
    # Convert AFN to AFD
    print("\n=== Converting AFN to AFD ===")
    afd = convert_afn_to_afd(afn, debug=args.debug, debug_file=debug_file)
    
    # Print AFD results
    print("\n=== AFD Result ===")
    afd.print_afd(use_aliases=True)
    
    print("\n=== AFD Transition Table ===")
    afd.print_transition_table()
    
    # Write output to file if specified
    if args.output:
        try:
            with open(args.output, 'w',encoding='utf-8') as out_file:
                out_file.write(afd.to_markdown())
            print(f"\nAFD result written to {args.output}")
        except Exception as e:
            print(f"Error writing output file {args.output}: {e}")
    
    # Close debug file if it was opened
    if debug_file:
        debug_file.close()
        print(f"Debug information written to {args.debug_output}")


if __name__ == "__main__":
    # If no arguments are provided, run with the hardcoded example
    if len(sys.argv) == 1:
        sys.argv.append('--example')
    main()