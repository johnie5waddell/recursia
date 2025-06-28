import cmd
import os


class RecursiaREPL(cmd.Cmd):
    """
    Read-Eval-Print Loop for the Recursia language. Provides an interactive
    command-line interface for executing Recursia code.
    """
    
    intro = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║                                                                   ║
    ║          Welcome to Recursia - The Organic Simulation Language    ║
    ║                                                                   ║
    ║          Type 'help' for a list of commands                       ║
    ║          To exit, type 'exit' or press Ctrl-D                     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    prompt = 'recursia> '
    
    def __init__(self, interpreter):
        super().__init__()
        self.interpreter = interpreter
        self.multiline_mode = False
        self.multiline_code = []
        self.history_file = os.path.expanduser('~/.recursia_history')
        
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def default(self, line):
        """Default action for any input not matching a command"""
        if self.multiline_mode:
            # In multiline mode, add each line to the buffer
            if line.strip() == '':
                # Empty line in multiline mode
                self.multiline_code.append('')
            elif line.strip() == 'end':
                # End multiline mode and execute the code
                code = '\nfrom typing import List\n'.join(self.multiline_code)
                self.multiline_mode = False
                self.multiline_code = []
                self._execute_code(code)
                return
            else:
                # Add the line to the buffer
                self.multiline_code.append(line)
        else:
            # Not in multiline mode, treat as code to execute
            self._execute_code(line)
    
    def _execute_code(self, code):
        """Execute Recursia code and handle the result"""
        # Skip empty code
        if not code.strip():
            return
        
        # Execute the code
        success, result, errors = self.interpreter.interpret(code)
        
        if not success:
            print("Execution failed:")
            for error in errors:
                print(f"  {error}")
        else:
            # Show execution time
            if result and 'execution_time' in result:
                print(f"Execution time: {result['execution_time']:.4f} seconds")
            
            # Update visualization
            self._show_visualization()
    
    def _show_visualization(self):
        """Show the current visualization"""
        # The interpreter already updates visualization components, 
        # but for REPL we can optionally draw a dashboard here
        if self.interpreter.state.visualization:
            self.interpreter.state.visualization['dashboard'].render_dashboard()
    
    def do_multiline(self, arg):
        """
        Enter multiline code mode
        Usage: multiline
        """
        print("Entering multiline mode. Type 'end' on a line by itself to execute.")
        self.multiline_mode = True
        self.multiline_code = []
    
    def do_load(self, arg):
        """
        Load and execute a Recursia file
        Usage: load <filename>
        """
        if not arg:
            print("Please specify a filename")
            return
        
        success, result, errors = self.interpreter.interpret_file(arg)
        
        if not success:
            print(f"Failed to execute {arg}:")
            for error in errors:
                print(f"  {error}")
        else:
            print(f"Successfully executed {arg}")
            if result and 'execution_time' in result:
                print(f"Execution time: {result['execution_time']:.4f} seconds")
            
            # Update visualization
            self._show_visualization()
    
    def do_run(self, arg):
        """
        Alias for 'load'
        Usage: run <filename>
        """
        self.do_load(arg)
    
    def do_debug(self, arg):
        """
        Toggle debug mode
        Usage: debug [on|off]
        """
        if not arg or arg.lower() not in ('on', 'off'):
            current = "on" if self.interpreter.state.debug_mode else "off"
            print(f"Debug mode is currently {current}")
            print("Usage: debug [on|off]")
            return
        
        self.interpreter.state.debug_mode = arg.lower() == 'on'
        print(f"Debug mode {'enabled' if self.interpreter.state.debug_mode else 'disabled'}")
    
    def do_trace(self, arg):
        """
        Toggle trace mode (prints each statement as it's executed)
        Usage: trace [on|off]
        """
        if not arg or arg.lower() not in ('on', 'off'):
            current = "on" if self.interpreter.state.trace_mode else "off"
            print(f"Trace mode is currently {current}")
            print("Usage: trace [on|off]")
            return
        
        self.interpreter.state.trace_mode = arg.lower() == 'on'
        print(f"Trace mode {'enabled' if self.interpreter.state.trace_mode else 'disabled'}")
    
    def do_step(self, arg):
        """
        Toggle step mode (pauses after each statement)
        Usage: step [on|off]
        """
        if not arg or arg.lower() not in ('on', 'off'):
            current = "on" if self.interpreter.state.step_mode else "off"
            print(f"Step mode is currently {current}")
            print("Usage: step [on|off]")
            return
        
        self.interpreter.state.step_mode = arg.lower() == 'on'
        print(f"Step mode {'enabled' if self.interpreter.state.step_mode else 'disabled'}")
    
    def do_breakpoint(self, arg):
        """
        Set or clear breakpoints
        Usage: breakpoint [set <line> | clear <line> | clear all | list]
        """
        parts = arg.split()
        if not parts or parts[0] not in ('set', 'clear', 'list'):
            print("Usage: breakpoint [set <line> | clear <line> | clear all | list]")
            return
        
        command = parts[0]
        
        if command == 'list':
            if not self.interpreter.state.breakpoints:
                print("No breakpoints set")
            else:
                print("Breakpoints:")
                for bp in sorted(self.interpreter.state.breakpoints):
                    print(f"  Line {bp}")
            return
        
        if command == 'set':
            if len(parts) < 2:
                print("Please specify a line number")
                return
            
            try:
                line = int(parts[1])
                self.interpreter.state.breakpoints.add(line)
                print(f"Breakpoint set at line {line}")
            except ValueError:
                print(f"Invalid line number: {parts[1]}")
            return
        
        if command == 'clear':
            if len(parts) < 2:
                print("Please specify a line number or 'all'")
                return
            
            if parts[1] == 'all':
                self.interpreter.state.breakpoints.clear()
                print("All breakpoints cleared")
                return
            
            try:
                line = int(parts[1])
                if line in self.interpreter.state.breakpoints:
                    self.interpreter.state.breakpoints.remove(line)
                    print(f"Breakpoint cleared at line {line}")
                else:
                    print(f"No breakpoint at line {line}")
            except ValueError:
                print(f"Invalid line number: {parts[1]}")
    
    def do_visualize(self, arg):
        """
        Visualize a quantum state or observer
        Usage: visualize [state <name> | observer <name> | field | all]
        """
        parts = arg.split()
        
        if not parts or parts[0] not in ('state', 'observer', 'field', 'all'):
            print("Usage: visualize [state <name> | observer <name> | field | all]")
            return
        
        command = parts[0]
        
        if command == 'state':
            if len(parts) < 2:
                print("Please specify a state name")
                return
            
            state_name = parts[1]
            if state_name not in self.interpreter.state.quantum_states:
                print(f"Unknown state: {state_name}")
                return
            
            state = self.interpreter.state.quantum_states[state_name]
            
            # Convert to state data if it's a quantum register
            if hasattr(state, 'get_state_vector'):
                state_data = {
                    'state_vector': state.get_state_vector(),
                    'coherence': state.coherence,
                    'entropy': state.entropy,
                    'is_entangled': state.is_entangled,
                    'entangled_with': state.entangled_with
                }
            else:
                state_data = state
            
            if self.interpreter.state.visualization:
                self.interpreter.state.visualization['field'].visualize_state(state_name, state_data)
            return
        
        if command == 'observer':
            if len(parts) < 2:
                print("Please specify an observer name")
                return
            
            observer_name = parts[1]
            if observer_name not in self.interpreter.state.observers:
                print(f"Unknown observer: {observer_name}")
                return
            
            observer_data = self.interpreter.state.observers[observer_name]
            
            if self.interpreter.state.visualization:
                self.interpreter.state.visualization['observer'].visualize_observer(observer_name, observer_data)
            return
        
        if command == 'field':
            field_data = {
                'states': self.interpreter.state.quantum_states
            }
            
            if self.interpreter.state.visualization:
                self.interpreter.state.visualization['field'].visualize_field(field_data)
            return
        
        if command == 'all':
            # Show dashboard with all data
            if self.interpreter.state.visualization:
                self.interpreter.state.visualization['dashboard'].render_dashboard()
            return
    
    def do_export(self, arg):
        """
        Export the current state to a file
        Usage: export [filename]
        """
        success = self.interpreter.state.export_state(arg if arg else None)
        
        if success:
            print(f"State exported to {arg if arg else 'recursia_state_<timestamp>.json'}")
        else:
            print("Failed to export state")
    
    def do_import(self, arg):
        """
        Import state from a file
        Usage: import <filename>
        """
        if not arg:
            print("Please specify a filename")
            return
        
        success = self.interpreter.state.import_state(arg)
        
        if success:
            print(f"State imported from {arg}")
            
            # Update visualization
            self._show_visualization()
        else:
            print(f"Failed to import state from {arg}")
    
    def do_reset(self, arg):
        """
        Reset the interpreter state
        Usage: reset
        """
        self.interpreter.state.reset()
        print("Interpreter state reset")
    
    def do_hardware(self, arg):
        """
        Configure quantum hardware backend
        Usage: hardware [connect [provider [device]] | disconnect | status]
        """
        parts = arg.split()
        
        if not parts:
            print("Usage: hardware [connect [provider [device]] | disconnect | status]")
            return
        
        command = parts[0]
        
        if command == 'connect':
            provider = parts[1] if len(parts) > 1 else 'auto'
            device = parts[2] if len(parts) > 2 else 'auto'
            
            success = self.interpreter.state.connect_hardware_backend(provider, device)
            
            if success:
                print(f"Connected to quantum hardware: {provider}, {device}")
                self.interpreter.use_hardware = True
            else:
                print("Failed to connect to quantum hardware")
            return
        
        if command == 'disconnect':
            if self.interpreter.state.hardware_backend:
                if hasattr(self.interpreter.state.hardware_backend, 'disconnect'):
                    self.interpreter.state.hardware_backend.disconnect()
                self.interpreter.state.hardware_backend = None
                self.interpreter.use_hardware = False
                print("Disconnected from quantum hardware")
            else:
                print("Not connected to quantum hardware")
            return
        
        if command == 'status':
            if self.interpreter.state.hardware_backend:
                connected = getattr(self.interpreter.state.hardware_backend, 'connected', False)
                provider = getattr(self.interpreter.state.hardware_backend, 'provider', 'unknown')
                device = getattr(self.interpreter.state.hardware_backend, 'device', 'unknown')
                
                print(f"Hardware backend: {provider}, {device}")
                print(f"Connected: {connected}")
                print(f"In use: {self.interpreter.use_hardware}")
                
                # Show available qubits if connected
                if connected:
                    available_qubits = getattr(self.interpreter.state.hardware_backend, 'available_qubits', 0)
                    print(f"Available qubits: {available_qubits}")
            else:
                print("No hardware backend configured")
            return
    
    def do_list(self, arg):
        """
        List states, observers, or variables
        Usage: list [states | observers | variables | all]
        """
        if not arg or arg == 'all':
            self.do_list('states')
            self.do_list('observers')
            self.do_list('variables')
            return
        
        if arg == 'states':
            if not self.interpreter.state.quantum_states:
                print("No quantum states defined")
            else:
                print("Quantum States:")
                for name, state in sorted(self.interpreter.state.quantum_states.items()):
                    if hasattr(state, 'num_qubits'):
                        print(f"  {name}: {state.num_qubits} qubits, coherence={getattr(state, 'coherence', 0):.2f}")
                    else:
                        print(f"  {name}")
            return
        
        if arg == 'observers':
            if not self.interpreter.state.observers:
                print("No observers defined")
            else:
                print("Observers:")
                for name, observer in sorted(self.interpreter.state.observers.items()):
                    focus = observer.get('focus', 'none')
                    print(f"  {name}: focus={focus}")
            return
        
        if arg == 'variables':
            if not self.interpreter.state.variables:
                print("No variables defined")
            else:
                print("Variables:")
                for name, value in sorted(self.interpreter.state.variables.items()):
                    # Limit long string values
                    if isinstance(value, str) and len(value) > 50:
                        value_str = f"{value[:47]}..."
                    else:
                        value_str = str(value)
                    print(f"  {name} = {value_str}")
            return
    
    def do_history(self, arg):
        """
        Show command history
        Usage: history [<count>]
        """
        pass
    def do_clear(self, arg):
        """
        Clear the screen
        Usage: clear
        """
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def do_info(self, arg):
        """
        Show information about the interpreter
        Usage: info
        """
        state_summary = self.interpreter.state.get_state_summary()
        
        print("Recursia Interpreter Information:")
        print(f"  Variables: {state_summary['variables']}")
        print(f"  Quantum States: {state_summary['quantum_states']}")
        print(f"  Observers: {state_summary['observers']}")
        print(f"  Execution Count: {state_summary['execution_count']}")
        print(f"  Debug Mode: {'enabled' if self.interpreter.state.debug_mode else 'disabled'}")
        print(f"  Trace Mode: {'enabled' if self.interpreter.state.trace_mode else 'disabled'}")
        print(f"  Step Mode: {'enabled' if self.interpreter.state.step_mode else 'disabled'}")
        print(f"  Hardware Backend: {'enabled' if self.interpreter.use_hardware else 'disabled'}")
       
    def do_exit(self, arg):
        """
        Exit the REPL
        Usage: exit
        """
        print("Exiting Recursia REPL")
        return True
    
    def do_quit(self, arg):
        """
        Alias for 'exit'
        Usage: quit
        """
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Handle Ctrl-D"""
        print()  # Print newline after Ctrl-D
        return self.do_exit(arg)
    
    def do_help(self, arg):
        """
        Show help message
        Usage: help [command]
        """
        if not arg:
            print("Recursia Command List:")
            print("  multiline        - Enter multiline code mode")
            print("  load <file>      - Load and execute a Recursia file")
            print("  run <file>       - Alias for 'load'")
            print("  debug [on|off]   - Toggle debug mode")
            print("  trace [on|off]   - Toggle trace mode")
            print("  step [on|off]    - Toggle step mode")
            print("  breakpoint       - Set or clear breakpoints")
            print("  visualize        - Visualize quantum states or observers")
            print("  export [file]    - Export state to a file")
            print("  import <file>    - Import state from a file")
            print("  reset            - Reset interpreter state")
            print("  hardware         - Configure quantum hardware backend")
            print("  list             - List states, observers, or variables")
            print("  history [count]  - Show command history")
            print("  clear            - Clear the screen")
            print("  info             - Show interpreter information")
            print("  help [command]   - Show help for a command")
            print("  exit, quit       - Exit the REPL")
            print()
            print("Enter Recursia code directly to execute it")
            print("Use 'help <command>' for detailed help on a command")
        else:
            # Call the parent class's do_help to show detailed help for a command
            super().do_help(arg)