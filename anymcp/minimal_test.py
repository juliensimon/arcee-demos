import os
import sys
import time
import logging
from mcp_use import MCPClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Create MCPClient from configuration dictionary
        config_path = os.path.join("config.json")
        logger.info(f"Loading config from {config_path}")
        
        mcp_client = MCPClient.from_config_file(config_path)
        logger.info("MCPClient created successfully")
        
        # Check if client was created properly
        logger.info(f"MCPClient: {mcp_client}")
        
        # Try to get the client's connectors
        if hasattr(mcp_client, "connectors"):
            logger.info(f"Client has {len(mcp_client.connectors)} connectors")
            for i, connector in enumerate(mcp_client.connectors):
                logger.info(f"Connector {i}: {connector}")
        else:
            logger.info("Client does not have 'connectors' attribute")
            
        # Try to get servers from config
        if hasattr(mcp_client, "config"):
            if hasattr(mcp_client.config, "mcpServers"):
                logger.info(f"Config has {len(mcp_client.config.mcpServers)} servers")
                for server_name, server_config in mcp_client.config.mcpServers.items():
                    logger.info(f"Server: {server_name}, Config: {server_config}")
            else:
                logger.info("Config does not have 'mcpServers' attribute")
        else:
            logger.info("Client does not have 'config' attribute")
        
        # Print all attributes
        logger.info("All client attributes:")
        for attr in dir(mcp_client):
            if not attr.startswith('__'):
                try:
                    value = getattr(mcp_client, attr)
                    logger.info(f"  {attr}: {type(value)}")
                except Exception as e:
                    logger.info(f"  {attr}: [Error accessing: {str(e)}]")
                    
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    logger.info("Starting minimal test")
    exit_code = main()
    logger.info(f"Test completed with exit code {exit_code}")
    sys.exit(exit_code) 