#!/usr/bin/env python3
"""
Simple web server for Porter.AI monitoring dashboard
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Set
import websockets
from aiohttp import web
import aiohttp_cors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardServer:
    """Simple web server with WebSocket support"""
    
    def __init__(self, host='localhost', http_port=8000, ws_port=8001):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP routes"""
        # Serve the dashboard HTML
        self.app.router.add_get('/', self.serve_dashboard)
        self.app.router.add_get('/dashboard', self.serve_dashboard)
        
        # Serve screenshot images (if needed)
        self.app.router.add_get('/screenshot/{filename}', self.serve_screenshot)
        
        # Add CORS support
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
            
    async def serve_dashboard(self, request):
        """Serve the dashboard HTML file"""
        dashboard_path = Path(__file__).parent.parent / 'frontend' / 'index.html'
        if dashboard_path.exists():
            with open(dashboard_path, 'r') as f:
                content = f.read()
            return web.Response(text=content, content_type='text/html')
        else:
            return web.Response(text='Dashboard not found', status=404)
            
    async def serve_screenshot(self, request):
        """Serve screenshot images"""
        filename = request.match_info['filename']
        screenshot_path = Path(__file__).parent / 'screenshots' / filename
        
        if screenshot_path.exists() and screenshot_path.is_file():
            with open(screenshot_path, 'rb') as f:
                content = f.read()
            return web.Response(body=content, content_type='image/png')
        else:
            return web.Response(text='Screenshot not found', status=404)
            
    async def websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        logger.info(f"New WebSocket connection")
        self.websocket_clients.add(websocket)
        
        try:
            # Send initial status
            await websocket.send(json.dumps({
                'type': 'status',
                'data': {
                    'connected': True,
                    'listening': False,
                    'speaking': False
                }
            }))
            
            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed")
        finally:
            self.websocket_clients.remove(websocket)
            
    async def handle_client_message(self, websocket, data):
        """Handle messages from dashboard client"""
        # For now, just echo back or handle control messages
        if data.get('type') == 'ping':
            await websocket.send(json.dumps({'type': 'pong'}))
            
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if self.websocket_clients:
            # Create tasks for all clients
            tasks = []
            for client in self.websocket_clients.copy():
                tasks.append(self.send_to_client(client, message))
            
            # Send to all clients concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def send_to_client(self, client, message):
        """Send message to a single client"""
        try:
            await client.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            # Client disconnected, remove from set
            if client in self.websocket_clients:
                self.websocket_clients.remove(client)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            
    async def send_screen_event(self, timestamp, description, importance, spoken=None, 
                               screenshot=None, tier='routine', ttl=None, has_screenshot=False):
        """Send screen event to dashboard"""
        logger.info(f"Broadcasting to {len(self.websocket_clients)} clients")
        await self.broadcast({
            'type': 'screen_event',
            'data': {
                'timestamp': timestamp,
                'description': description,
                'importance': importance,
                'spoken': spoken,
                'screenshot': screenshot,
                'tier': tier,
                'ttl': ttl,
                'has_screenshot': has_screenshot
            }
        })
        
    async def send_voice_interaction(self, timestamp, user_said, ai_response):
        """Send voice interaction to dashboard"""
        await self.broadcast({
            'type': 'voice_interaction',
            'data': {
                'timestamp': timestamp,
                'user_said': user_said,
                'ai_response': ai_response
            }
        })
        
    async def update_status(self, listening=False, speaking=False):
        """Update system status indicators"""
        await self.broadcast({
            'type': 'status',
            'data': {
                'listening': listening,
                'speaking': speaking
            }
        })
        
    async def start(self):
        """Start both HTTP and WebSocket servers"""
        # Start WebSocket server
        ws_server = await websockets.serve(
            self.websocket_handler,
            self.host,
            self.ws_port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.ws_port}")
        
        # Start HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.http_port)
        await site.start()
        logger.info(f"HTTP server started on http://{self.host}:{self.http_port}")
        
        return ws_server

async def main():
    """Standalone server for testing"""
    server = DashboardServer()
    await server.start()
    
    logger.info("Dashboard server running. Press Ctrl+C to stop.")
    logger.info(f"Open http://localhost:8000 in your browser")
    
    # Keep server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down server...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass