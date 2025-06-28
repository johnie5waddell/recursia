#!/usr/bin/env python3
import asyncio
import websockets
import json
import time

async def test_metrics():
    uri = "ws://localhost:8080/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Track metrics over time
        last_metrics = {}
        message_count = 0
        
        while message_count < 20:  # Collect 20 messages
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'metrics':
                    message_count += 1
                    metrics = data['data']
                    
                    # Show key metrics and their changes
                    print(f"\n=== Message {message_count} ===")
                    print(f"RSP: {metrics['rsp']:.4f} (Δ: {metrics['rsp'] - last_metrics.get('rsp', metrics['rsp']):.4f})")
                    print(f"Coherence: {metrics['coherence']:.4f} (Δ: {metrics['coherence'] - last_metrics.get('coherence', metrics['coherence']):.4f})")
                    print(f"Entropy: {metrics['entropy']:.4f} (Δ: {metrics['entropy'] - last_metrics.get('entropy', metrics['entropy']):.4f})")
                    print(f"Strain: {metrics['strain']:.4f} (Δ: {metrics['strain'] - last_metrics.get('strain', metrics['strain']):.4f})")
                    print(f"Focus: {metrics['focus']:.4f}")
                    print(f"Derivatives - drsp_dt: {metrics['drsp_dt']:.4f}, di_dt: {metrics['di_dt']:.4f}")
                    
                    last_metrics = metrics.copy()
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_metrics())