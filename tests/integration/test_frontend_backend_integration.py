"""
Integration tests between React frontend and Python backend.
Tests WebSocket communication, data flow, and real-time synchronization.
"""

import pytest
import asyncio
import json
import websockets
import threading
import time
from typing import Dict, Any, List
import numpy as np

# Import backend components
from src.core.compiler import RecursiaCompiler
from src.core.interpreter import RecursiaInterpreter
from src.physics.physics_engine_proper import PhysicsEngine
from src.visualization.quantum_visualization_engine import QuantumVisualizationEngine

# Mock WebSocket server for testing
class MockWebSocketServer:
    """Mock WebSocket server for testing frontend-backend communication"""
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.message_log = []
        self.server = None
        self.physics_engine = PhysicsEngine()
        self.compiler = RecursiaCompiler()
        self.interpreter = RecursiaInterpreter()
        self.viz_engine = QuantumVisualizationEngine()
        
    async def register_client(self, websocket, path):
        """Register new client connection"""
        self.clients.add(websocket)
        try:
            await self.handle_client(websocket)
        finally:
            self.clients.remove(websocket)
    
    async def handle_client(self, websocket):
        """Handle client messages"""
        async for message in websocket:
            try:
                data = json.loads(message)
                self.message_log.append(data)
                response = await self.process_message(data)
                if response:
                    await websocket.send(json.dumps(response))
            except Exception as e:
                error_response = {
                    'type': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
                await websocket.send(json.dumps(error_response))
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response"""
        msg_type = message.get('type')
        
        if msg_type == 'execute_code':
            return await self.handle_code_execution(message)
        elif msg_type == 'simulation_start':
            return await self.handle_simulation_start(message)
        elif msg_type == 'simulation_pause':
            return await self.handle_simulation_pause(message)
        elif msg_type == 'simulation_stop':
            return await self.handle_simulation_stop(message)
        elif msg_type == 'parameter_change':
            return await self.handle_parameter_change(message)
        elif msg_type == 'temporal_navigation':
            return await self.handle_temporal_navigation(message)
        elif msg_type == 'export_data':
            return await self.handle_data_export(message)
        else:
            return {'type': 'error', 'error': f'Unknown message type: {msg_type}'}
    
    async def handle_code_execution(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code execution request"""
        code = message.get('payload', {}).get('code', '')
        target = message.get('payload', {}).get('target', 'quantum_simulator')
        
        try:
            # Compile code
            tokens = self.compiler.lexer.tokenize(code)
            ast = self.compiler.parser.parse(tokens)
            semantic_result = self.compiler.semantic_analyzer.analyze(ast)
            
            if not semantic_result.is_valid:
                return {
                    'type': 'compilation_result',
                    'success': False,
                    'errors': semantic_result.errors,
                    'warnings': semantic_result.warnings
                }
            
            compiled_result = self.compiler.compile(ast, target=target)
            
            if not compiled_result.success:
                return {
                    'type': 'compilation_result',
                    'success': False,
                    'errors': compiled_result.errors
                }
            
            # Execute code
            execution_result = self.interpreter.execute(compiled_result.bytecode, self.physics_engine)
            
            if execution_result.success:
                # Generate visualization data
                viz_data = self.viz_engine.generate_complete_scene_visualization(self.physics_engine)
                
                return {
                    'type': 'compilation_result',
                    'success': True,
                    'execution_success': True,
                    'visualization_data': viz_data,
                    'simulation_state': self.get_simulation_state()
                }
            else:
                return {
                    'type': 'compilation_result',
                    'success': True,
                    'execution_success': False,
                    'errors': execution_result.errors
                }
                
        except Exception as e:
            return {
                'type': 'compilation_result',
                'success': False,
                'errors': [str(e)]
            }
    
    async def handle_simulation_start(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simulation start request"""
        self.physics_engine.start_simulation()
        return {
            'type': 'simulation_update',
            'data': self.get_simulation_state()
        }
    
    async def handle_simulation_pause(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simulation pause request"""
        self.physics_engine.pause_simulation()
        return {
            'type': 'simulation_update',
            'data': self.get_simulation_state()
        }
    
    async def handle_simulation_stop(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simulation stop request"""
        self.physics_engine.stop_simulation()
        return {
            'type': 'simulation_update',
            'data': self.get_simulation_state()
        }
    
    async def handle_parameter_change(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parameter change request"""
        parameter = message.get('payload', {}).get('parameter')
        value = message.get('payload', {}).get('value')
        
        self.physics_engine.set_parameter(parameter, value)
        
        return {
            'type': 'parameter_update',
            'parameter': parameter,
            'value': value,
            'simulation_state': self.get_simulation_state()
        }
    
    async def handle_temporal_navigation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle temporal navigation request"""
        target_time = message.get('payload', {}).get('time', 0)
        
        self.physics_engine.set_simulation_time(target_time)
        
        return {
            'type': 'temporal_update',
            'current_time': target_time,
            'simulation_state': self.get_simulation_state()
        }
    
    async def handle_data_export(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data export request"""
        export_format = message.get('payload', {}).get('format', 'json')
        
        export_data = self.physics_engine.export_data(format=export_format)
        
        return {
            'type': 'export_complete',
            'format': export_format,
            'data': export_data,
            'timestamp': time.time()
        }
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            'is_running': self.physics_engine.is_running(),
            'is_paused': self.physics_engine.is_paused(),
            'current_time': self.physics_engine.get_current_time(),
            'total_time': self.physics_engine.get_total_time(),
            'quantum_states': [state.to_dict() for state in self.physics_engine.get_quantum_states()],
            'observers': [obs.to_dict() for obs in self.physics_engine.get_observers()],
            'fields': [field.to_dict() for field in self.physics_engine.get_fields()],
            'metrics': self.physics_engine.get_metrics()
        }
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(self.register_client, self.host, self.port)
        return self.server
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

class TestFrontendBackendIntegration:
    """Integration tests between frontend and backend"""
    
    @pytest.fixture
    async def mock_server(self):
        """Create mock WebSocket server for testing"""
        server = MockWebSocketServer()
        await server.start_server()
        yield server
        await server.stop_server()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, mock_server):
        """Test basic WebSocket connection"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        async with websockets.connect(uri) as websocket:
            # Send ping message
            ping_message = {
                'type': 'ping',
                'timestamp': time.time()
            }
            await websocket.send(json.dumps(ping_message))
            
            # Should not crash
            assert len(mock_server.clients) == 1
    
    @pytest.mark.asyncio
    async def test_code_execution_workflow(self, mock_server):
        """Test complete code execution workflow"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        # Simple Recursia program
        test_code = """
        quantum_state test_qubit {
            qubits: 1
            coherence: 1.0
        }
        
        observer test_observer {
            type: conscious
            awareness: 0.8
        }
        
        result = measure(test_qubit, test_observer)
        visualize(test_qubit, "3d_probability")
        """
        
        async with websockets.connect(uri) as websocket:
            # Send code execution request
            execute_message = {
                'type': 'execute_code',
                'payload': {
                    'code': test_code,
                    'target': 'quantum_simulator',
                    'optimization_level': 'O2'
                }
            }
            
            await websocket.send(json.dumps(execute_message))
            
            # Wait for response
            response_data = await websocket.recv()
            response = json.loads(response_data)
            
            # Verify response structure
            assert response['type'] == 'compilation_result'
            assert response['success'] == True
            assert 'simulation_state' in response
            assert 'visualization_data' in response
            
            # Verify simulation state
            sim_state = response['simulation_state']
            assert len(sim_state['quantum_states']) == 1
            assert len(sim_state['observers']) == 1
            assert sim_state['quantum_states'][0]['name'] == 'test_qubit'
            assert sim_state['observers'][0]['name'] == 'test_observer'
    
    @pytest.mark.asyncio
    async def test_simulation_control_workflow(self, mock_server):
        """Test simulation control commands"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        async with websockets.connect(uri) as websocket:
            # Start simulation
            start_message = {'type': 'simulation_start'}
            await websocket.send(json.dumps(start_message))
            
            start_response = json.loads(await websocket.recv())
            assert start_response['type'] == 'simulation_update'
            assert start_response['data']['is_running'] == True
            
            # Pause simulation
            pause_message = {'type': 'simulation_pause'}
            await websocket.send(json.dumps(pause_message))
            
            pause_response = json.loads(await websocket.recv())
            assert pause_response['type'] == 'simulation_update'
            assert pause_response['data']['is_paused'] == True
            
            # Stop simulation
            stop_message = {'type': 'simulation_stop'}
            await websocket.send(json.dumps(stop_message))
            
            stop_response = json.loads(await websocket.recv())
            assert stop_response['type'] == 'simulation_update'
            assert stop_response['data']['is_running'] == False
    
    @pytest.mark.asyncio
    async def test_parameter_synchronization(self, mock_server):
        """Test parameter changes synchronization"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        async with websockets.connect(uri) as websocket:
            # Change quantum parameter
            param_message = {
                'type': 'parameter_change',
                'payload': {
                    'parameter': 'quantum.coherence_threshold',
                    'value': 0.85
                }
            }
            
            await websocket.send(json.dumps(param_message))
            
            response = json.loads(await websocket.recv())
            assert response['type'] == 'parameter_update'
            assert response['parameter'] == 'quantum.coherence_threshold'
            assert response['value'] == 0.85
    
    @pytest.mark.asyncio
    async def test_real_time_visualization_updates(self, mock_server):
        """Test real-time visualization data updates"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        # Create quantum system first
        setup_code = """
        quantum_state dynamic_qubit {
            qubits: 1
            coherence: 1.0
        }
        
        observer conscious_obs {
            type: conscious
            awareness: 0.9
        }
        """
        
        async with websockets.connect(uri) as websocket:
            # Execute setup code
            execute_message = {
                'type': 'execute_code',
                'payload': {'code': setup_code, 'target': 'quantum_simulator'}
            }
            await websocket.send(json.dumps(execute_message))
            setup_response = json.loads(await websocket.recv())
            assert setup_response['success'] == True
            
            # Start simulation
            await websocket.send(json.dumps({'type': 'simulation_start'}))
            await websocket.recv()  # Consume start response
            
            # Test temporal navigation
            for target_time in [1.0, 2.0, 3.0, 4.0, 5.0]:
                temporal_message = {
                    'type': 'temporal_navigation',
                    'payload': {'time': target_time}
                }
                await websocket.send(json.dumps(temporal_message))
                
                response = json.loads(await websocket.recv())
                assert response['type'] == 'temporal_update'
                assert abs(response['current_time'] - target_time) < 0.01
                
                # Verify simulation state is updated
                sim_state = response['simulation_state']
                assert 'quantum_states' in sim_state
                assert 'observers' in sim_state
    
    @pytest.mark.asyncio
    async def test_complex_quantum_simulation(self, mock_server):
        """Test complex quantum simulation with entanglement"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        entanglement_code = """
        quantum_state alice {
            qubits: 1
            coherence: 1.0
            position: [-2, 0, 0]
        }
        
        quantum_state bob {
            qubits: 1
            coherence: 1.0
            position: [2, 0, 0]
        }
        
        entangle(alice, bob)
        
        observer alice_obs {
            type: conscious
            awareness: 0.8
            position: [-3, 0, 0]
        }
        
        observer bob_obs {
            type: conscious
            awareness: 0.8
            position: [3, 0, 0]
        }
        
        visualize([alice, bob], "entanglement_network")
        """
        
        async with websockets.connect(uri) as websocket:
            execute_message = {
                'type': 'execute_code',
                'payload': {'code': entanglement_code, 'target': 'quantum_simulator'}
            }
            await websocket.send(json.dumps(execute_message))
            
            response = json.loads(await websocket.recv())
            assert response['success'] == True
            
            # Verify entangled system
            sim_state = response['simulation_state']
            assert len(sim_state['quantum_states']) == 2
            assert len(sim_state['observers']) == 2
            
            # Check visualization data
            viz_data = response['visualization_data']
            assert 'entanglement_connections' in viz_data
            assert len(viz_data['entanglement_connections']) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_server):
        """Test error handling and system recovery"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        # Invalid Recursia code
        invalid_code = """
        quantum_state invalid {
            qubits: "not_a_number"
            invalid_property: true
        }
        
        measure(nonexistent_state, nonexistent_observer)
        """
        
        async with websockets.connect(uri) as websocket:
            execute_message = {
                'type': 'execute_code',
                'payload': {'code': invalid_code, 'target': 'quantum_simulator'}
            }
            await websocket.send(json.dumps(execute_message))
            
            response = json.loads(await websocket.recv())
            assert response['success'] == False
            assert 'errors' in response
            assert len(response['errors']) > 0
            
            # System should still be responsive after error
            ping_message = {'type': 'simulation_start'}
            await websocket.send(json.dumps(ping_message))
            
            recovery_response = json.loads(await websocket.recv())
            assert recovery_response['type'] == 'simulation_update'
    
    @pytest.mark.asyncio
    async def test_data_export_functionality(self, mock_server):
        """Test data export functionality"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        # Create simulation data first
        setup_code = """
        quantum_state export_test {
            qubits: 2
            coherence: 0.9
        }
        
        observer export_observer {
            type: conscious
            awareness: 0.7
        }
        
        measure(export_test, export_observer)
        """
        
        async with websockets.connect(uri) as websocket:
            # Setup simulation
            execute_message = {
                'type': 'execute_code',
                'payload': {'code': setup_code, 'target': 'quantum_simulator'}
            }
            await websocket.send(json.dumps(execute_message))
            await websocket.recv()  # Consume response
            
            # Test different export formats
            export_formats = ['json', 'csv', 'hdf5']
            
            for format_type in export_formats:
                export_message = {
                    'type': 'export_data',
                    'payload': {
                        'format': format_type,
                        'timestamp': time.time()
                    }
                }
                await websocket.send(json.dumps(export_message))
                
                response = json.loads(await websocket.recv())
                assert response['type'] == 'export_complete'
                assert response['format'] == format_type
                assert 'data' in response
                assert 'timestamp' in response
    
    @pytest.mark.asyncio
    async def test_concurrent_client_handling(self, mock_server):
        """Test handling multiple concurrent clients"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        # Connect multiple clients
        clients = []
        for i in range(5):
            client = await websockets.connect(uri)
            clients.append(client)
        
        assert len(mock_server.clients) == 5
        
        # Send messages from all clients simultaneously
        tasks = []
        for i, client in enumerate(clients):
            message = {
                'type': 'execute_code',
                'payload': {
                    'code': f'quantum_state client_{i}_state {{ qubits: 1 }}',
                    'target': 'quantum_simulator'
                }
            }
            task = asyncio.create_task(client.send(json.dumps(message)))
            tasks.append(task)
        
        # Wait for all sends to complete
        await asyncio.gather(*tasks)
        
        # Collect responses from all clients
        response_tasks = []
        for client in clients:
            response_tasks.append(asyncio.create_task(client.recv()))
        
        responses = await asyncio.gather(*response_tasks)
        
        # Verify all clients got responses
        assert len(responses) == 5
        for response_data in responses:
            response = json.loads(response_data)
            assert response['type'] == 'compilation_result'
        
        # Clean up connections
        for client in clients:
            await client.close()

class TestPerformanceIntegration:
    """Performance tests for frontend-backend integration"""
    
    @pytest.mark.asyncio
    async def test_high_frequency_updates(self, mock_server):
        """Test system performance with high-frequency updates"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        async with websockets.connect(uri) as websocket:
            # Send rapid parameter changes
            start_time = time.time()
            update_count = 100
            
            for i in range(update_count):
                param_message = {
                    'type': 'parameter_change',
                    'payload': {
                        'parameter': 'quantum.coherence_threshold',
                        'value': 0.5 + (i % 50) / 100.0
                    }
                }
                await websocket.send(json.dumps(param_message))
                await websocket.recv()  # Consume response
            
            total_time = time.time() - start_time
            updates_per_second = update_count / total_time
            
            # Should handle at least 10 updates per second
            assert updates_per_second > 10.0, f"Update rate too low: {updates_per_second:.1f}/s"
    
    @pytest.mark.asyncio
    async def test_large_data_transfer(self, mock_server):
        """Test transfer of large simulation datasets"""
        uri = f"ws://{mock_server.host}:{mock_server.port}"
        
        # Create large quantum system
        large_system_code = """
        memory_field large_memory {
            dimensions: [64, 64, 64]
            resolution: 0.05
        }
        """ + "\n".join([
            f"""
            quantum_state large_state_{i} {{
                qubits: 2
                coherence: {0.5 + i / 200.0}
                position: [{i % 8}, {(i // 8) % 8}, {i // 64}]
            }}
            """ for i in range(100)
        ])
        
        async with websockets.connect(uri) as websocket:
            start_time = time.time()
            
            execute_message = {
                'type': 'execute_code',
                'payload': {'code': large_system_code, 'target': 'quantum_simulator'}
            }
            await websocket.send(json.dumps(execute_message))
            
            response_data = await websocket.recv()
            response = json.loads(response_data)
            
            transfer_time = time.time() - start_time
            data_size = len(response_data.encode('utf-8')) / 1024 / 1024  # MB
            
            assert response['success'] == True
            assert len(response['simulation_state']['quantum_states']) == 100
            
            # Should handle reasonable data transfer rates
            transfer_rate = data_size / transfer_time  # MB/s
            assert transfer_rate > 1.0, f"Data transfer rate too low: {transfer_rate:.2f} MB/s"

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])