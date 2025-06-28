from typing import List, Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy operations if needed
    class _NumpyFallback:
        # Add ndarray type for compatibility
        ndarray = list
        def array(self, x): return x
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def ones(self, shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()

from src.quantum.quantum_register import QuantumRegister


class QuantumHardwareBackend:
    """
    Interface to real quantum hardware for Recursia
    
    This class provides a common interface to connect Recursia programs
    to actual quantum hardware devices from providers like IBM, Rigetti, etc.
    
    Note: To use this class, you need to install the appropriate quantum SDK:
    - For IBM: pip install qiskit qiskit-ibmq-provider
    - For Rigetti: pip install pyquil
    - For IonQ: pip install ionq
    - For Google: pip install cirq
    """
    
    def __init__(self, provider: str = "auto", device: str = "auto", credentials: dict = None, options: dict = None):
        """
        Initialize the quantum hardware backend
        
        Args:
            provider: Quantum hardware provider ("ibm", "rigetti", "ionq", "auto")
            device: Specific quantum device to use
            credentials: Authentication credentials for the provider
            options: Additional options for the backend
        """
        self.provider = provider
        self.device = device
        self.credentials = credentials or {}
        self.options = options or {}
        self.connected = False
        self.registers = {}
        self.compiled_circuits = {}
        self.provider_backend = None
        
        # Track available qubits
        self.available_qubits = 0
        self.connectivity_map = {}  # Maps which qubits can interact with which
        
        # Cache for results
        self.last_results = {}
        
        # Check for required libraries
        self._available_providers = self._check_available_providers()
        
        if not self._available_providers:
            self.logger.warning("No quantum hardware providers available.")
            self.logger.info("To use real quantum hardware, install one of the following packages:")
            self.logger.info("  - IBM: pip install qiskit qiskit-ibmq-provider")
            self.logger.info("  - Rigetti: pip install pyquil")
            self.logger.info("  - Google: pip install cirq")
            self.logger.info("  - IonQ: pip install ionq")
        else:
            self.logger.info(f"Available quantum hardware providers: {', '.join(self._available_providers)}")
    
    def _check_available_providers(self):
        """Check which quantum providers are available based on installed packages"""
        available = []
        
        # Check for IBM Qiskit
        try:
            import qiskit # type: ignore
            available.append("ibm")
        except ImportError:
            pass
        
        # Check for Rigetti PyQuil
        try:
            import pyquil # type: ignore
            available.append("rigetti")
        except ImportError:
            pass
        
        # Check for Google Cirq
        try:
            import cirq # type: ignore
            available.append("google")
        except ImportError:
            pass
        
        # Check for IonQ (once it becomes available)
        try:
            import ionq # type: ignore
            available.append("ionq")
        except ImportError:
            pass
        
        return available
    
    def connect(self) -> bool:
        """
        Connect to the quantum hardware provider
        
        Returns:
            bool: True if connection was successful
        """
        if not self._available_providers:
            self.logger.warning("No quantum hardware providers available. Using simulator instead.")
            return False
        
        try:
            if self.provider == "auto":
                # Try to connect to available providers
                for provider in self._available_providers:
                    if self._try_connect_provider(provider):
                        self.provider = provider
                        self.connected = True
                        break
            elif self.provider in self._available_providers:
                # Connect to the specified provider
                self.connected = self._try_connect_provider(self.provider)
            else:
                self.logger.error(f"Provider '{self.provider}' is not available.")
                self.logger.info(f"Available providers: {', '.join(self._available_providers)}")
                self.logger.info("Install the required package or use 'auto' to try available providers.")
                return False
            
            if self.connected:
                # Get hardware specifications
                self._fetch_device_specs()
                return True
            else:
                self.logger.warning(f"Failed to connect to quantum hardware provider: {self.provider}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error connecting to quantum hardware: {e}")
            return False
    
    def _try_connect_provider(self, provider: str) -> bool:
        """
        Try to connect to a specific provider
        
        Args:
            provider: Provider name
            
        Returns:
            bool: True if connection was successful
        """
        if provider == "ibm":
            # IBM Quantum Experience / Qiskit
            try:
                from qiskit import IBMQ # type: ignore
                
                # Get API token from credentials or environment
                import os
                token = self.credentials.get("ibm_token", os.environ.get("IBM_QUANTUM_TOKEN"))
                
                if not token:
                    self.logger.error("IBM Quantum token not found in credentials or environment")
                    self.logger.info("Please provide a token via:")
                    self.logger.info("  - credentials={'ibm_token': 'your_token'}")
                    self.logger.info("  - Setting IBM_QUANTUM_TOKEN environment variable")
                    return False
                
                # Load account from token
                IBMQ.save_account(token, overwrite=True)
                IBMQ.load_account()
                
                # Get provider
                provider_obj = IBMQ.get_provider()
                
                # Select device
                if self.device == "auto":
                    # Get least busy backend with at least 5 qubits
                    from qiskit.providers.ibmq  import least_busy # type: ignore
                    large_enough_devices = provider_obj.backends(
                        filters=lambda x: x.configuration().n_qubits >= 5 and 
                                          not x.configuration().simulator
                    )
                    if large_enough_devices:
                        self.provider_backend = least_busy(large_enough_devices)
                    else:
                        # Fall back to simulator if no hardware available
                        self.provider_backend = provider_obj.get_backend('ibmq_qasm_simulator')
                else:
                    self.provider_backend = provider_obj.get_backend(self.device)
                
                self.logger.info(f"Connected to IBM Quantum device: {self.provider_backend.name()}")
                return True
            
            except Exception as e:
                self.logger.error(f"Error connecting to IBM Quantum: {e}")
                return False
        
        elif provider == "rigetti":
            # Rigetti Forest / PyQuil
            try:
                from pyquil import get_qc # type: ignore
                
                # If device not specified, use the 8Q-Agave QPU simulator
                device_name = self.device if self.device != "auto" else "8q-agave-qvm"
                
                # Connect to the Quantum Computer
                self.provider_backend = get_qc(device_name)
                
                self.logger.info(f"Connected to Rigetti device: {device_name}")
                return True
            
            except Exception as e:
                self.logger.error(f"Error connecting to Rigetti: {e}")
                return False
        
        elif provider == "google":
            # Google Cirq
            try:
                import cirq # type: ignore
                
                # Currently Cirq doesn't have direct access to Google hardware for most users
                # We'll create a simulator as a placeholder
                self.provider_backend = cirq.Simulator()
                
                self.logger.info("Connected to Google Cirq simulator (hardware access not yet implemented)")
                return True
            
            except Exception as e:
                self.logger.error(f"Error connecting to Google Cirq: {e}")
                return False
        
        elif provider == "ionq":
            # IonQ - placeholder
            try:
                # This would use the IonQ Python SDK when available
                self.logger.info("IonQ support is a placeholder for future implementation")
                return False
            
            except Exception as e:
                self.logger.error(f"Error connecting to IonQ: {e}")
                return False
        
        else:
            self.logger.error(f"Unsupported quantum hardware provider: {provider}")
            return False
        
    def _fetch_device_specs(self):
        """Fetch specifications of the connected quantum device"""
        if not self.connected or not self.provider_backend:
            return
        
        try:
            if self.provider == "ibm":
                # Get device configuration
                config = self.provider_backend.configuration()
                self.available_qubits = config.n_qubits
                
                # Get connectivity map
                if hasattr(config, 'coupling_map') and config.coupling_map:
                    for source, targets in enumerate(config.coupling_map):
                        self.connectivity_map[source] = targets
            
            elif self.provider == "rigetti":
                # For Rigetti, get the topology from the device
                topology = self.provider_backend.qc.topology
                self.available_qubits = len(topology.nodes)
                self.connectivity_map = {int(q1): [int(q2) for q2 in topology.neighbors(q1)] 
                                       for q1 in topology.nodes}
        
        except Exception as e:
            self.logger.error(f"Error fetching device specifications: {e}")
    
    def create_register(self, name: str, num_qubits: int) -> QuantumRegister:
        """
        Create a quantum register on the hardware
        
        Args:
            name: Register name
            num_qubits: Number of qubits
            
        Returns:
            QuantumRegister: The created register
        """
        if not self.connected:
            self.logger.error("Not connected to quantum hardware. Call connect() first.")
            raise ValueError("Not connected to quantum hardware")
        
        if num_qubits > self.available_qubits:
            self.logger.error(f"Requested {num_qubits} qubits, but only {self.available_qubits} available")
            raise ValueError(f"Requested {num_qubits} qubits, but only {self.available_qubits} available")
        
        # Create a virtual register for tracking
        register = QuantumRegister(num_qubits, name)
        self.registers[name] = register
        
        return register
    
    def apply_gate(self, register_name: str, gate_name: str,
                  target_qubits: Union[int, List[int]],
                  control_qubits: Optional[Union[int, List[int]]] = None,
                  params: Optional[List[float]] = None) -> bool:
        """
        Apply a quantum gate on the hardware
        
        Args:
            register_name: Register name
            gate_name: Gate name
            target_qubits: Target qubit index or indices
            control_qubits: Optional control qubit index or indices
            params: Optional gate parameters
            
        Returns:
            bool: True if gate was applied successfully
        """
        if not self.connected:
            self.logger.error("Not connected to quantum hardware. Call connect() first.")
            return False
        
        register = self.registers.get(register_name)
        if not register:
            self.logger.error(f"Unknown register: {register_name}")
            return False
        
        # First, update the simulator version for tracking
        success = register.apply_gate(gate_name, target_qubits, control_qubits, params)
        if not success:
            return False
        
        # For real hardware, we would compile and queue the operation here
        # For now, we'll just track the operations to build a circuit later
        
        # Create circuit for this register if it doesn't exist
        if register_name not in self.compiled_circuits:
            if self.provider == "ibm":
                # Create Qiskit circuit
                try:
                    from qiskit import QuantumCircuit # type: ignore
                    self.compiled_circuits[register_name] = QuantumCircuit(register.num_qubits, register.num_qubits)
                except ImportError:
                    self.logger.error("Failed to create Qiskit circuit. Make sure qiskit is installed.")
                    return False
            elif self.provider == "rigetti":
                # Create PyQuil program
                try:
                    from pyquil import Program # type: ignore
                    self.compiled_circuits[register_name] = Program()
                except ImportError:
                    self.logger.error("Failed to create PyQuil program. Make sure pyquil is installed.")
                    return False
            elif self.provider == "google":
                # Create Cirq circuit
                try:
                    import cirq # type: ignore
                    self.compiled_circuits[register_name] = cirq.Circuit()
                except ImportError:
                    self.logger.error("Failed to create Cirq circuit. Make sure cirq is installed.")
                    return False
        
        # Add gate to circuit based on provider
        circuit = self.compiled_circuits[register_name]
        
        # Convert target and control qubits to lists if they aren't already
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        if control_qubits is not None and isinstance(control_qubits, int):
            control_qubits = [control_qubits]
        
        try:
            if self.provider == "ibm":
                self._add_gate_to_qiskit_circuit(circuit, gate_name, target_qubits, control_qubits, params)
            elif self.provider == "rigetti":
                self._add_gate_to_pyquil_program(circuit, gate_name, target_qubits, control_qubits, params)
            elif self.provider == "google":
                self._add_gate_to_cirq_circuit(circuit, gate_name, target_qubits, control_qubits, params)
            else:
                self.logger.warning(f"Gate addition not implemented for provider: {self.provider}")
        except Exception as e:
            self.logger.error(f"Error adding gate to circuit: {e}")
            return False
        
        return True
    
    def _add_gate_to_qiskit_circuit(self, circuit, gate_name, target_qubits, control_qubits, params):
        """Add a gate to a Qiskit circuit"""
        if gate_name in ("H_gate", "Hadamard_gate"):
            for t in target_qubits:
                circuit.h(t)
        elif gate_name in ("X_gate", "PauliX_gate"):
            for t in target_qubits:
                circuit.x(t)
        elif gate_name in ("Y_gate", "PauliY_gate"):
            for t in target_qubits:
                circuit.y(t)
        elif gate_name in ("Z_gate", "PauliZ_gate"):
            for t in target_qubits:
                circuit.z(t)
        elif gate_name in ("CNOT_gate", "CX_gate"):
            if control_qubits and len(target_qubits) >= 1:
                circuit.cx(control_qubits[0], target_qubits[0])
        elif gate_name == "CZ_gate":
            if control_qubits and len(target_qubits) >= 1:
                circuit.cz(control_qubits[0], target_qubits[0])
        elif gate_name == "SWAP_gate":
            if len(target_qubits) >= 2:
                circuit.swap(target_qubits[0], target_qubits[1])
        elif gate_name == "S_gate":
            for t in target_qubits:
                circuit.s(t)
        elif gate_name == "T_gate":
            for t in target_qubits:
                circuit.t(t)
        elif gate_name == "RX_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    circuit.rx(params[0], t)
        elif gate_name == "RY_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    circuit.ry(params[0], t)
        elif gate_name == "RZ_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    circuit.rz(params[0], t)
        elif gate_name == "PhaseShift_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    circuit.p(params[0], t)
        elif gate_name == "Toffoli_gate" or gate_name == "CCNOT_gate":
            if control_qubits and len(control_qubits) >= 2 and len(target_qubits) >= 1:
                circuit.ccx(control_qubits[0], control_qubits[1], target_qubits[0])
        elif gate_name == "QFT_gate":
            # Import QFT circuit
            from qiskit.circuit.library import QFT # type: ignore
            qft = QFT(len(target_qubits))
            circuit.append(qft, target_qubits)
        elif gate_name == "InverseQFT_gate":
            # Import QFT circuit and invert it
            from qiskit.circuit.library import QFT # type: ignore
            iqft = QFT(len(target_qubits)).inverse()
            circuit.append(iqft, target_qubits)
        else:
            raise ValueError(f"Unsupported gate for IBM Qiskit: {gate_name}")
    
    def _add_gate_to_pyquil_program(self, program, gate_name, target_qubits, control_qubits, params):
        """Add a gate to a PyQuil program"""
        from pyquil import gates # type: ignore
        
        if gate_name in ("H_gate", "Hadamard_gate"):
            for t in target_qubits:
                program += gates.H(t)
        elif gate_name in ("X_gate", "PauliX_gate"):
            for t in target_qubits:
                program += gates.X(t)
        elif gate_name in ("Y_gate", "PauliY_gate"):
            for t in target_qubits:
                program += gates.Y(t)
        elif gate_name in ("Z_gate", "PauliZ_gate"):
            for t in target_qubits:
                program += gates.Z(t)
        elif gate_name in ("CNOT_gate", "CX_gate"):
            if control_qubits and len(target_qubits) >= 1:
                program += gates.CNOT(control_qubits[0], target_qubits[0])
        elif gate_name == "CZ_gate":
            if control_qubits and len(target_qubits) >= 1:
                program += gates.CZ(control_qubits[0], target_qubits[0])
        elif gate_name == "SWAP_gate":
            if len(target_qubits) >= 2:
                program += gates.SWAP(target_qubits[0], target_qubits[1])
        elif gate_name == "S_gate":
            for t in target_qubits:
                program += gates.S(t)
        elif gate_name == "T_gate":
            for t in target_qubits:
                program += gates.T(t)
        elif gate_name == "RX_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    program += gates.RX(params[0], t)
        elif gate_name == "RY_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    program += gates.RY(params[0], t)
        elif gate_name == "RZ_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    program += gates.RZ(params[0], t)
        elif gate_name == "PhaseShift_gate":
            if params and len(params) >= 1:
                for t in target_qubits:
                    program += gates.PHASE(params[0], t)
        elif gate_name == "Toffoli_gate" or gate_name == "CCNOT_gate":
            if control_qubits and len(control_qubits) >= 2 and len(target_qubits) >= 1:
                program += gates.CCNOT(control_qubits[0], control_qubits[1], target_qubits[0])
        else:
            raise ValueError(f"Unsupported gate for Rigetti PyQuil: {gate_name}")
    
    def _add_gate_to_cirq_circuit(self, circuit, gate_name, target_qubits, control_qubits, params):
        """Add a gate to a Cirq circuit"""
        import cirq # type: ignore
        
        # Convert indices to Cirq qubits
        cirq_qubits = [cirq.LineQubit(q) for q in target_qubits]
        cirq_control_qubits = [cirq.LineQubit(q) for q in control_qubits] if control_qubits else []
        
        if gate_name in ("H_gate", "Hadamard_gate"):
            for q in cirq_qubits:
                circuit.append(cirq.H(q))
        elif gate_name in ("X_gate", "PauliX_gate"):
            for q in cirq_qubits:
                circuit.append(cirq.X(q))
        elif gate_name in ("Y_gate", "PauliY_gate"):
            for q in cirq_qubits:
                circuit.append(cirq.Y(q))
        elif gate_name in ("Z_gate", "PauliZ_gate"):
            for q in cirq_qubits:
                circuit.append(cirq.Z(q))
        elif gate_name in ("CNOT_gate", "CX_gate"):
            if cirq_control_qubits and cirq_qubits:
                circuit.append(cirq.CNOT(cirq_control_qubits[0], cirq_qubits[0]))
        elif gate_name == "CZ_gate":
            if cirq_control_qubits and cirq_qubits:
                circuit.append(cirq.CZ(cirq_control_qubits[0], cirq_qubits[0]))
        elif gate_name == "SWAP_gate":
            if len(cirq_qubits) >= 2:
                circuit.append(cirq.SWAP(cirq_qubits[0], cirq_qubits[1]))
        elif gate_name == "S_gate":
            for q in cirq_qubits:
                circuit.append(cirq.S(q))
        elif gate_name == "T_gate":
            for q in cirq_qubits:
                circuit.append(cirq.T(q))
        elif gate_name == "RX_gate":
            if params and len(params) >= 1:
                for q in cirq_qubits:
                    circuit.append(cirq.rx(params[0])(q))
        elif gate_name == "RY_gate":
            if params and len(params) >= 1:
                for q in cirq_qubits:
                    circuit.append(cirq.ry(params[0])(q))
        elif gate_name == "RZ_gate":
            if params and len(params) >= 1:
                for q in cirq_qubits:
                    circuit.append(cirq.rz(params[0])(q))
        elif gate_name == "PhaseShift_gate":
            if params and len(params) >= 1:
                for q in cirq_qubits:
                    circuit.append(cirq.ZPowGate(exponent=params[0]/(2*np.pi))(q))
        elif gate_name == "Toffoli_gate" or gate_name == "CCNOT_gate":
            if len(cirq_control_qubits) >= 2 and cirq_qubits:
                circuit.append(cirq.TOFFOLI(cirq_control_qubits[0], cirq_control_qubits[1], cirq_qubits[0]))
        else:
            raise ValueError(f"Unsupported gate for Google Cirq: {gate_name}")
    
    def measure(self, register_name: str, qubits=None, basis="computational_basis", shots=1024):
        """
        Measure qubits on the hardware
        
        Args:
            register_name: Register name
            qubits: Qubit indices to measure (None means all)
            basis: Measurement basis
            shots: Number of measurement shots
            
        Returns:
            dict: Measurement results
        """
        if not self.connected:
            self.logger.error("Not connected to quantum hardware. Call connect() first.")
            return {"error": "Not connected to quantum hardware"}
                   
        if register_name not in self.registers:
            print(f"Unknown register: {register_name}")
            return None
        
        register = self.registers[register_name]
        circuit = register['circuit']
        
        if circuit is None:
            print("Circuit creation failed")
            return None
        
        # If no qubits specified, measure all
        if qubits is None:
            qubits = list(range(register['num_qubits']))
        elif not isinstance(qubits, list):
            qubits = [qubits]
        
        # Map logical qubits to physical qubits
        physical_qubits = [register['physical_qubits'][q] for q in qubits if q < register['num_qubits']]
        
        # Handle different measurement bases
        if basis != "computational_basis" and basis != "Z_basis":
            # Apply basis change before measurement
            if basis == "X_basis":
                # Apply Hadamard gates to change to X basis
                for q in physical_qubits:
                    self._apply_gate_ibm(circuit, "H_gate", [q], None, None) if self.provider == "ibm" else \
                    self._apply_gate_rigetti(circuit, "H_gate", [q], None, None)
            
            elif basis == "Y_basis":
                # Apply S† then H to change to Y basis
                for q in physical_qubits:
                    # Apply S† (conjugate of S gate)
                    if self.provider == "ibm":
                        circuit.sdg(q)
                    else:
                        # For other providers, implement S† as RZ(-π/2)
                        self._apply_gate_rigetti(circuit, "RZ_gate", [q], None, [-np.pi/2])
                    
                    # Apply Hadamard
                    self._apply_gate_ibm(circuit, "H_gate", [q], None, None) if self.provider == "ibm" else \
                    self._apply_gate_rigetti(circuit, "H_gate", [q], None, None)
        
        # Add measurement operations to the circuit
        try:
            if self.provider == "ibm":
                # For IBM, add measurement operations to the circuit
                for i, q in enumerate(physical_qubits):
                    circuit.measure(q, i)  # Measure qubit q into classical bit i
                
                # Execute the circuit
                from qiskit import execute # type: ignore
                job = execute(circuit, self.provider_backend, shots=shots)
                result = job.result()
                
                # Process results
                counts = result.get_counts(circuit)
                
                # Convert to our format
                measurement_result = {
                    'outcome': max(counts, key=counts.get),  # Most frequent result
                    'probabilities': {outcome: count/shots for outcome, count in counts.items()},
                    'shots': shots
                }
                
                # Cache the result
                self.last_results[register_name] = measurement_result
                
                return measurement_result
            
            elif self.provider == "rigetti":
                # For Rigetti, add measurement operations and classical registers
                from pyquil import gates # type: ignore
                
                # Create classical registers for each qubit to be measured
                classical_regs = [f"ro[{i}]" for i in range(len(physical_qubits))]
                
                # Add MEASURE instructions
                for i, q in enumerate(physical_qubits):
                    circuit += gates.MEASURE(q, classical_regs[i])
                
                # Compile and run the program
                executable = self.provider_backend.compile(circuit)
                results = self.provider_backend.run(executable, shots=shots)
                
                # Process results (specific to Rigetti's format)
                # This would need to be adjusted based on the actual API
                outcome_counts = {}
                for trial in results:
                    outcome = ''.join(str(bit) for bit in trial)
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                
                # Convert to our format
                measurement_result = {
                    'outcome': max(outcome_counts, key=outcome_counts.get),  # Most frequent result
                    'probabilities': {outcome: count/shots for outcome, count in outcome_counts.items()},
                    'shots': shots
                }
                
                # Cache the result
                self.last_results[register_name] = measurement_result
                
                return measurement_result
        
        except Exception as e:
            print(f"Error measuring qubits: {e}")
            return None
    
    def entangle(self, register1_name, register2_name, register1_qubits=None, register2_qubits=None, method="direct"):
        """
        Entangle qubits between two registers
        
        Args:
            register1_name: First register name
            register2_name: Second register name
            register1_qubits: Qubits in first register
            register2_qubits: Qubits in second register
            method: Entanglement method
            
        Returns:
            bool: True if entanglement was successful
        """
        # For hardware implementation, we'd need to combine the two registers into a single circuit
        # and apply entangling gates between them
        print("Hardware entanglement between registers is not fully implemented yet")
        return False
    
    def teleport(self, source_name, destination_name, source_qubit=0, destination_qubit=0):
        """
        Teleport a qubit from one register to another
        
        Args:
            source_name: Source register name
            destination_name: Destination register name
            source_qubit: Source qubit index
            destination_qubit: Destination qubit index
            
        Returns:
            bool: True if teleportation was successful
        """
        # For hardware implementation, we'd need to implement the teleportation protocol
        # with proper entanglement, measurements, and classical communication
        print("Hardware teleportation is not fully implemented yet")
        return False
    
    def reset(self, register_name, qubits=None):
        """
        Reset qubits to |0> state
        
        Args:
            register_name: Register name
            qubits: Qubit indices to reset
            
        Returns:
            bool: True if reset was successful
        """
        if not self.connected:
            print("Not connected to quantum hardware. Call connect() first.")
            return False
        
        if register_name not in self.registers:
            print(f"Unknown register: {register_name}")
            return False
        
        register = self.registers[register_name]
        
        # If no qubits specified, reset the entire register
        if qubits is None:
            # Create a new empty circuit
            register['circuit'] = self._create_empty_circuit(register['num_qubits'])
            return True
        
        # For specific qubits, we need to measure them and conditionally apply X gates
        # This is a reset protocol for quantum hardware
        
        # Map logical qubits to physical qubits
        if not isinstance(qubits, list):
            qubits = [qubits]
            
        physical_qubits = [register['physical_qubits'][q] for q in qubits if q < register['num_qubits']]
        
        try:
            if self.provider == "ibm":
                # For IBM Qiskit, use the built-in reset operation
                for q in physical_qubits:
                    register['circuit'].reset(q)
                return True
            
            elif self.provider == "rigetti":
                # For Rigetti, implement reset as measure and conditional X
                from pyquil import gates # type: ignore
                from pyquil.quil import Pragma # type: ignore
                
                for q in physical_qubits:
                    # Add pragmas to ensure reset works
                    register['circuit'] += Pragma('PRESERVE_BLOCK')
                    
                    # Measure qubit to a classical register
                    classical_reg = f"ro[{q}]"
                    register['circuit'] += gates.MEASURE(q, classical_reg)
                    
                    # If measurement result is 1, apply X gate to flip it back to 0
                    register['circuit'] += gates.X(q).controlled(classical_reg)
                    
                    register['circuit'] += Pragma('END_PRESERVE_BLOCK')
                
                return True
            
            # Implement for other providers as needed
            
            return True
        
        except Exception as e:
            print(f"Error resetting qubits: {e}")
            return False
    
    def get_register_properties(self, register_name):
        """
        Get properties of a register
        
        Args:
            register_name: Register name
            
        Returns:
            dict: Register properties
        """
        if not self.connected:
            print("Not connected to quantum hardware. Call connect() first.")
            return None
        
        if register_name not in self.registers:
            print(f"Unknown register: {register_name}")
            return None
        
        register = self.registers[register_name]
        
        # Return basic properties
        # For real hardware, we would query the device for more detailed information
        return {
            'num_qubits': register['num_qubits'],
            'physical_qubits': register['physical_qubits'],
            'provider': self.provider,
            'device': self.device
        }
    
    def disconnect(self):
        """Disconnect from the quantum hardware provider"""
        if self.connected:
            # Close connection and clean up resources
            self.connected = False
            self.provider_backend = None
            print(f"Disconnected from {self.provider} quantum hardware")